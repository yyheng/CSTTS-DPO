"""CMIspeech critic: Whisper-LAL pseudo frame language labels -> code-mixing index per wav.

Env: --ckpt is either the authors' whole-pickled WhisperWithLAL (loads only in the env that made the
pickle; on the cluster ``whisperold``: torch 1.13.1, transformers 4.38.0, openai-whisper 20250625) or a
lal/export_state_dict.py state dict (any env with torch, transformers and openai-whisper, e.g. the main
env). Runs on a GPU compute node (falls back to CPU with a warning when no GPU is visible). Never on
the login node.

Inputs (at least one):
  --cands_tsv  one or more cands.tsv from 10_gen_candidates.py (columns utt cand wav ...)
  --manifest   manifest TSV from 00_prep_manifest.py (or any TSV with utt + wav [+ text_tts]);
               its wav column is the ground-truth audio and is written with cand='gt'.
  --wav_scp    a Kaldi dir (or its wav.scp): devman/devsge themselves or a 20_synthesize.py output;
               rows are written with cand=--cand_label (default 'gt', as in 11_/12_ --wav_scp mode).
               Use e.g. --cand_label synth together with --manifest to get gt + synthetic rows (and
               the mean ΔCMI of --print_summary) in one file.
Rows are grouped per utterance (gt first, then candidates in file order); --shard/--nshards and
--limit operate on utterances so that an utterance's gt and candidate rows stay in one file.
When candidate rows (cands.tsv, or wav.scp with a non-'gt' label) are present, gt rows are only
scored for utterances that have a candidate (ΔCMI needs nothing else); --gt_all keeps every gt row.

Per row: 16 kHz audio -> 80-mel log spectrogram padded to 30 s -> encoder -> language_cls ->
argmax over {0 zh, 1 en, 2 blank, 3 other}; the first ceil(dur / 0.02) frames (<= 1500) are
kept; CMI = (T - max_k T_k) / T over zh+en frames (or all four with --count_all_classes).

Output TSV columns (exactly):  utt cand wav n_frames n_zh n_en n_blank n_other cmi [text_cmi] [labels_rle]
  text_cmi (only with --text_cmi, which requires --manifest) is the token-level CMI of the
  manifest text_tts (common.text_cmi: CJK-char token = zh, ASCII-letter token = en), the same
  value for every row of an utterance; empty when the utterance has no text. Default columns
  are unchanged without the flag.
  labels_rle (with --dump_labels) is a run-length encoding like '0x120,1x35,2x10'.
CMI is written in [0, 1]; --print_summary prints per-cand means (and mean ΔCMI vs gt) in %.
Rows that could not be scored (unreadable file, non-finite samples, ...) are logged, listed in
<out>.failed.txt (rewritten on every run: `utt cand wav reason`) and skipped; re-run with --resume
to retry only those. --resume also repairs an --out whose last line was cut short by a hard kill
(common.repair_torn_tsv: the partial line is truncated away and its row scored again).
Exit status (shared critic policy, same flags in 11_/12_): 0 when the failed fraction
(failed / rows in scope) is <= --max_fail_frac (default 0.01; a WARNING is logged and the
afterok chain / .done marker proceed); 2 when it is above that fraction, or with --strict and
any failure at all.

Paths: --ckpt defaults to the configured CMI_DPO_LAL_CKPT (cmi_dpo.paths, config/paths.env or the
environment; the WhisperLAL module is found via CMI_DPO_LAL_CODE_DIR), resolved after parsing so
that --help works without the env file; --show_paths prints the configured paths and exits.

Example:
  python -u scripts/13_score_cmi.py \\
      --cands_tsv exp/round1/cands.tsv \\
      --manifest data/train.tsv \\
      --out exp/round1/cmi.tsv --batch 8 --print_summary
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import OrderedDict
from typing import Any, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmi_dpo import common, lal_cmi, paths  # noqa: E402

BASE_COLUMNS = ["utt", "cand", "wav", "n_frames", "n_zh", "n_en", "n_blank", "n_other", "cmi"]
TEXT_CMI_COLUMN = "text_cmi"
LABELS_COLUMN = "labels_rle"
GT_CAND = "gt"
LOG = logging.getLogger("score_cmi")


class ShowPathsAction(argparse.Action):
    """--show_paths: print cmi_dpo.paths.describe() and exit, before the required flags are checked."""

    def __init__(self, option_strings: list[str], dest: str, **kwargs: Any) -> None:
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS,
                         help=kwargs.get("help", "print the configured paths (cmi_dpo.paths) and exit"))

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
                 values: Any, option_string: str | None = None) -> None:
        print(paths.describe())
        parser.exit()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score CMIspeech (Whisper-LAL frame language labels) for candidate and/or ground-truth "
                    "wavs (GPU job; env: the one that made a pickled checkpoint, or any env for an exported state dict).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--cands_tsv", nargs="+", default=None,
                   help="cands.tsv file(s) from 10_gen_candidates.py (columns utt cand wav ...).")
    p.add_argument("--manifest", default=None,
                   help="Manifest TSV (utt + wav [+ text_tts]); its wav rows are scored with cand='gt'.")
    p.add_argument("--wav_scp", default=None,
                   help="Kaldi data dir (or path to its wav.scp), e.g. devman or a 20_synthesize.py output; "
                        "its rows are scored with cand=--cand_label.")
    p.add_argument("--cand_label", default=GT_CAND,
                   help="cand value written for --wav_scp rows ('gt' = ground truth; anything else = a candidate).")
    p.add_argument("--gt_all", action="store_true",
                   help="Score every ground-truth row even when candidates are given (default: only the utterances "
                        "that have at least one candidate row).")
    p.add_argument("--ckpt", default=None,
                   help="WhisperWithLAL checkpoint (whole pickle or exported state_dict, see cmi_dpo.lal_cmi); "
                        "default: the configured CMI_DPO_LAL_CKPT.")
    p.add_argument("--out", required=True,
                   help="Output TSV (columns: utt cand wav n_frames n_zh n_en n_blank n_other cmi [text_cmi] "
                        "[labels_rle]).")
    p.add_argument("--batch", type=int, default=8, help="Utterances per encoder forward (mels batched on GPU).")
    p.add_argument("--device", default="cuda:0", help="torch device; falls back to cpu when CUDA is unavailable.")
    p.add_argument("--dump_labels", action="store_true",
                   help="Add a labels_rle column with the run-length encoded frame labels, e.g. '0x120,1x35,2x10'.")
    p.add_argument("--count_all_classes", action="store_true",
                   help="Count blank/other frames in T(u) too (default: only zh + en frames).")
    p.add_argument("--limit", type=int, default=0, help="Only score the first N utterances (all their rows); 0 = all.")
    p.add_argument("--shard", type=int, default=0, help="Shard index in [0, nshards) over utterances (round-robin).")
    p.add_argument("--nshards", type=int, default=1, help="Number of shards.")
    p.add_argument("--resume", action="store_true",
                   help="Append to an existing --out, skipping (utt,cand) rows already scored "
                        "(a torn last line is repaired first).")
    p.add_argument("--print_summary", action="store_true",
                   help="After scoring, print mean CMI per cand group (and mean ΔCMI vs gt) in %% from --out.")
    p.add_argument("--text_cmi", action="store_true",
                   help="Add a text_cmi column (token-level CMI of the manifest text_tts, common.text_cmi); "
                        "requires --manifest. Default columns are unchanged without it.")
    p.add_argument("--max_fail_frac", type=float, default=0.01,
                   help="Exit 0 (with a WARNING) when failed rows / rows in scope <= this fraction; exit 2 above it.")
    p.add_argument("--strict", action="store_true", help="Exit 2 on ANY failed row (ignores --max_fail_frac).")
    p.add_argument("--show_paths", action=ShowPathsAction)
    return p


# ---------------------------------------------------------------------------
# Row collection
# ---------------------------------------------------------------------------
def _require_columns(path: str, rows: list[dict], needed: tuple[str, ...]) -> None:
    if rows and any(c not in rows[0] for c in needed):
        raise ValueError(f"{path}: missing column(s) {[c for c in needed if c not in rows[0]]}")


def resolve_kaldi_dir(path: str) -> str:
    """Accept either a Kaldi dir or a path to its wav.scp (same rule as 11_/12_)."""
    if os.path.isfile(path):
        return os.path.dirname(os.path.abspath(path))
    return path


def collect_rows(cands_tsvs: Optional[list[str]], manifest: Optional[str], wav_scp: Optional[str] = None,
                 cand_label: str = GT_CAND, gt_all: bool = False) -> tuple[list[dict], dict[str, str]]:
    """Merge gt + candidate rows grouped per utterance.

    Ground-truth rows come from ``manifest`` (cand='gt') and from ``wav_scp`` when
    ``cand_label == 'gt'``; candidate rows from ``cands_tsvs`` and from ``wav_scp`` with any
    other label. Unless ``gt_all``, gt rows are kept only for utterances that have a
    candidate row (when there are candidate rows at all).

    Returns ``(rows, text_by_utt)``: rows are dicts with utt/cand/wav, ordered by first
    appearance of the utterance with the gt row first, then candidates in file order;
    duplicates of an (utt, cand) key keep the first occurrence. ``text_by_utt`` holds the
    manifest ``text_tts`` (or strip_tags of the Kaldi ``text``) for the text-CMI summary.
    """
    per_utt: "OrderedDict[str, list[dict]]" = OrderedDict()
    seen: set[tuple[str, str]] = set()
    text_by_utt: dict[str, str] = {}
    n_dup = 0

    def add(utt: str, cand: str, wav: str) -> None:
        nonlocal n_dup
        key = (utt, cand)
        if key in seen:
            n_dup += 1
            return
        seen.add(key)
        per_utt.setdefault(utt, []).append({"utt": utt, "cand": cand, "wav": wav})

    gt_rows: list[tuple[str, str]] = []              # (utt, wav)
    cand_rows: list[tuple[str, str, str]] = []       # (utt, cand, wav)
    if manifest:
        mrows = common.read_tsv(manifest)
        _require_columns(manifest, mrows, ("utt", "wav"))
        for r in mrows:
            gt_rows.append((r["utt"], r["wav"]))
            if "text_tts" in r:
                text_by_utt[r["utt"]] = r["text_tts"]
        LOG.info("manifest %s: %d ground-truth rows", manifest, len(mrows))
    if wav_scp:
        kdir = resolve_kaldi_dir(wav_scp)
        kd = common.read_kaldi_dir(kdir)
        for utt, v in kd.items():
            if cand_label == GT_CAND:
                gt_rows.append((utt, v["wav"]))
            else:
                cand_rows.append((utt, cand_label, v["wav"]))
            text_by_utt.setdefault(utt, common.strip_tags(v["text_raw"]))
        LOG.info("wav.scp %s: %d rows with cand=%s", kdir, len(kd), cand_label)
    for path in cands_tsvs or []:
        crows = common.read_tsv(path)
        _require_columns(path, crows, ("utt", "cand", "wav"))
        for r in crows:
            if r["cand"] == GT_CAND:
                raise ValueError(f"{path}: candidate label '{GT_CAND}' is reserved for ground-truth rows")
            cand_rows.append((r["utt"], r["cand"], r["wav"]))
        LOG.info("cands %s: %d candidate rows", path, len(crows))

    cand_utts = {utt for utt, _, _ in cand_rows}
    n_no_cand = 0
    for utt, wav in gt_rows:
        if cand_rows and not gt_all and utt not in cand_utts:
            n_no_cand += 1
            continue
        add(utt, GT_CAND, wav)
    if n_no_cand:
        LOG.info("%d ground-truth rows skipped: their utterance has no candidate row (--gt_all keeps them)", n_no_cand)
    for utt, cand, wav in cand_rows:
        add(utt, cand, wav)
    if n_dup:
        LOG.warning("%d duplicate (utt, cand) rows ignored", n_dup)
    rows = [r for utt_rows in per_utt.values() for r in utt_rows]
    return rows, text_by_utt


def select_utts(rows: list[dict], shard: int, nshards: int, limit: int) -> list[dict]:
    """Keep the rows of shard ``shard``/``nshards`` of the utterances, then the first ``limit`` utterances."""
    utts = list(OrderedDict.fromkeys(r["utt"] for r in rows))
    utts = common.shard(utts, shard, nshards)
    if limit > 0:
        utts = utts[:limit]
    keep = set(utts)
    return [r for r in rows if r["utt"] in keep]


def truncate_partial_last_line(out: str, n_cols: int) -> int:
    """Thin wrapper over common.repair_torn_tsv (the logic moved there so 11_/12_ share it).

    Cuts off a last line that a hard kill left incomplete (no newline or wrong column count):
    dropping it while reading is not enough, the next append would be glued onto it. Returns
    the number of bytes truncated (0 = nothing to repair).
    """
    removed = common.repair_torn_tsv(out, n_cols)
    if removed:
        LOG.warning("resume: repaired torn tail of %s (%d bytes truncated)", out, removed)
    return removed


def read_done(out: str, columns: list[str]) -> set[tuple[str, str]]:
    """(utt, cand) keys already present in ``out``; the header must match ``columns``.

    A partial last line (interrupted append) is truncated away first so the file is safe to
    append to and its row is retried.
    """
    with open(out, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
    if header != columns:
        raise ValueError(f"{out}: existing columns {header} differ from {columns}; "
                         "run without --resume (or with matching --dump_labels / --text_cmi) to rewrite it")
    truncate_partial_last_line(out, len(columns))
    return {(r["utt"], r["cand"]) for r in common.read_tsv(out)}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_batch(model: torch.nn.Module, chunk: list[dict], langs: tuple[int, ...], dump_labels: bool,
                device: torch.device, text_by_utt: Optional[dict[str, str]] = None
                ) -> tuple[list[dict], list[tuple[dict, str]]]:
    """Score up to --batch rows in one encoder forward. Returns (output rows, failures).

    With ``text_by_utt`` (--text_cmi) every row also carries ``text_cmi`` = common.text_cmi of
    its utterance's text_tts ('' when the utterance has no text).
    """
    mels: list[torch.Tensor] = []
    durs: list[float] = []
    kept: list[dict] = []
    failures: list[tuple[dict, str]] = []
    for r in chunk:
        try:
            audio = common.load_audio16k(r["wav"])
            dur = len(audio) / float(lal_cmi.SAMPLE_RATE)
            if dur > lal_cmi.MAX_AUDIO_SEC:
                LOG.warning("%s cand=%s: %.2f s > 30 s, only the first 30 s are scored", r["utt"], r["cand"], dur)
                dur = lal_cmi.MAX_AUDIO_SEC
            elif len(audio) == 0:
                LOG.warning("%s cand=%s: empty audio (%s) -> 0 frames, CMI 0", r["utt"], r["cand"], r["wav"])
            # whisper_mel rejects NaN/Inf samples (a diverged vocoder output); one NaN would otherwise
            # poison the whole mel, and the argmax labels would silently yield a bogus CMI.
            mel = lal_cmi.whisper_mel(audio, device=device)
        except Exception as exc:  # noqa: BLE001 - unreadable/missing/non-finite file: report, keep going
            failures.append((r, f"{type(exc).__name__}: {exc}"))
            continue
        mels.append(mel)
        durs.append(dur)
        kept.append(r)
    if not kept:
        return [], failures

    labels = lal_cmi.frame_language_labels(model, torch.stack(mels), durs)
    out_rows: list[dict] = []
    for r, lab in zip(kept, labels):
        cmi, counts = lal_cmi.cmi_from_labels(lab, langs)
        row = {"utt": r["utt"], "cand": r["cand"], "wav": r["wav"], "n_frames": counts["n_frames"],
               "n_zh": counts["n_zh"], "n_en": counts["n_en"], "n_blank": counts["n_blank"],
               "n_other": counts["n_other"], "cmi": f"{cmi:.6f}"}
        if text_by_utt is not None:
            text = text_by_utt.get(r["utt"])
            row[TEXT_CMI_COLUMN] = f"{common.text_cmi(text):.6f}" if text is not None else ""
        if dump_labels:
            row[LABELS_COLUMN] = lal_cmi.labels_rle(lab)
        out_rows.append(row)
    return out_rows, failures


def score_rows(todo: list[dict], args: argparse.Namespace, columns: list[str], langs: tuple[int, ...],
               failed_txt: str, text_by_utt: Optional[dict[str, str]] = None) -> int:
    """Score all rows in batches, appending to args.out; failures go to ``failed_txt``; returns their count."""
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        LOG.warning("CUDA not available; scoring on cpu (slow)")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    model = lal_cmi.load_lal_model(args.ckpt, device)
    n_done = 0
    n_failed = 0
    n_batches = (len(todo) + args.batch - 1) // args.batch
    for b in range(n_batches):
        chunk = todo[b * args.batch: (b + 1) * args.batch]
        rows, failures = score_batch(model, chunk, langs, args.dump_labels, device, text_by_utt)
        for r, why in failures:
            LOG.error("FAILED %s cand=%s wav=%s: %s", r["utt"], r["cand"], r["wav"], why)
            common.append_failed(failed_txt, r["utt"], r["cand"], r["wav"], why)
        n_failed += len(failures)
        for row in rows:
            common.append_tsv(args.out, row, columns)
        n_done += len(rows)
        if (b + 1) % 50 == 0 or b + 1 == n_batches:
            LOG.info("scored %d/%d rows (%d failed)", n_done, len(todo), n_failed)
    return n_failed


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(out: str, text_by_utt: dict[str, str], utts_in_scope: set[str]) -> None:
    """Mean CMI per cand group, mean ΔCMI vs gt per candidate group, and text-CMI sanity (all in %)."""
    rows = common.read_tsv(out)
    cmi = {(r["utt"], r["cand"]): float(r["cmi"]) for r in rows}
    gt = {u: v for (u, c), v in cmi.items() if c == GT_CAND}
    groups: "OrderedDict[str, list[float]]" = OrderedDict()
    deltas: "OrderedDict[str, list[float]]" = OrderedDict()
    for r in rows:
        groups.setdefault(r["cand"], []).append(cmi[(r["utt"], r["cand"])])
        if r["cand"] != GT_CAND and r["utt"] in gt:
            deltas.setdefault(r["cand"], []).append(lal_cmi.delta_cmi(cmi[(r["utt"], r["cand"])], gt[r["utt"]]))
    print(f"== CMIspeech summary: {out} ({len(rows)} rows) ==")
    print(f"{'cand':>10} {'n':>8} {'mean_CMI%':>10} {'mean_dCMI%':>11} {'n_with_gt':>9}")
    all_c: list[float] = []
    all_d: list[float] = []
    for cand, vals in groups.items():
        d = deltas.get(cand, [])
        d_str = f"{100.0 * float(np.mean(d)):10.2f}" if d else f"{'-':>10}"
        print(f"{cand:>10} {len(vals):8d} {100.0 * float(np.mean(vals)):10.2f} {d_str:>11} {len(d):9d}")
        if cand != GT_CAND:
            all_c.extend(vals)
            all_d.extend(d)
    if all_c:
        d_str = f"{100.0 * float(np.mean(all_d)):10.2f}" if all_d else f"{'-':>10}"
        print(f"{'all_cands':>10} {len(all_c):8d} {100.0 * float(np.mean(all_c)):10.2f} {d_str:>11} {len(all_d):9d}")
    texts = [text_by_utt[u] for u in utts_in_scope if u in text_by_utt]
    if texts:
        mean_text = 100.0 * float(np.mean([common.text_cmi(t) for t in texts]))
        print(f"text CMI (manifest text_tts, token-level) mean over {len(texts)} utts: {mean_text:.2f}%")


# ---------------------------------------------------------------------------
def main() -> int:
    args = build_parser().parse_args()
    common.setup_logging()
    if not args.cands_tsv and not args.manifest and not args.wav_scp:
        raise SystemExit("at least one of --cands_tsv / --manifest / --wav_scp is required")
    if not args.cand_label:
        raise SystemExit("--cand_label must not be empty")
    if args.wav_scp and args.manifest and args.cand_label == GT_CAND:
        raise SystemExit(f"--wav_scp rows would also be cand='{GT_CAND}' and collide with the --manifest gt rows; "
                         "give the wav.scp set another --cand_label (e.g. synth)")
    if args.batch < 1:
        raise SystemExit("--batch must be >= 1")
    if args.text_cmi and not args.manifest:
        raise SystemExit("--text_cmi needs the manifest text_tts: pass --manifest")
    if not 0.0 <= args.max_fail_frac <= 1.0:
        raise SystemExit("--max_fail_frac must be in [0, 1]")
    if args.ckpt is None:  # resolved lazily so that --help works without config/paths.env
        args.ckpt = paths.lal_ckpt()
    columns = (BASE_COLUMNS + ([TEXT_CMI_COLUMN] if args.text_cmi else [])
               + ([LABELS_COLUMN] if args.dump_labels else []))
    langs: tuple[int, ...] = (lal_cmi.ZH, lal_cmi.EN, lal_cmi.BLANK, lal_cmi.OTHER) if args.count_all_classes \
        else (lal_cmi.ZH, lal_cmi.EN)
    LOG.info("CMI classes counted: %s", [lal_cmi.CLASS_NAMES[k] for k in langs])

    rows, text_by_utt = collect_rows(args.cands_tsv, args.manifest, args.wav_scp, args.cand_label, args.gt_all)
    rows = select_utts(rows, args.shard, args.nshards, args.limit)
    utts_in_scope = {r["utt"] for r in rows}
    LOG.info("shard %d/%d: %d rows over %d utterances", args.shard, args.nshards, len(rows), len(utts_in_scope))

    done: set[tuple[str, str]] = set()
    if args.resume and os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        done = read_done(args.out, columns)
        LOG.info("resume: %d rows already in %s", len(done), args.out)
    else:
        common.write_tsv(args.out, [], columns)
    todo = [r for r in rows if (r["utt"], r["cand"]) not in done]
    LOG.info("%d rows to score (%d skipped as done)", len(todo), len(rows) - len(todo))
    failed_txt = args.out + ".failed.txt"
    common.reset_failed(failed_txt)  # every row in scope is (re)attempted below, so the old list is stale

    n_failed = score_rows(todo, args, columns, langs, failed_txt,
                          text_by_utt if args.text_cmi else None) if todo else 0
    if args.print_summary:
        print_summary(args.out, text_by_utt, utts_in_scope)
    code = common.failure_exit_code(n_failed, len(rows), args.max_fail_frac, args.strict, failed_txt, LOG)
    if code == 0:
        LOG.info("done: %s (%d rows failed)", args.out, n_failed)
    return code


if __name__ == "__main__":
    sys.exit(main())
