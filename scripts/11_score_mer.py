"""MER critic: transcribe candidate (or real) wavs with the SEAME-fine-tuned Whisper and score MER.

Env: asr-whisper (torch 2.9, transformers 4.57, jiwer optional cross-check).
Runs on a GPU compute node only (never on the login node).

Decoding mirrors the authors' baselines run_whisper.py exactly: an HF
``pipeline('automatic-speech-recognition')`` with ``chunk_length_s=30`` fed a generator of
``{'array': float32 16 kHz, 'sampling_rate': 16000}`` dicts, ``generate_kwargs={'task': 'transcribe'}``
(plus ``language`` when --language is given); the pipeline preserves input order.

Two input modes (exactly one required):
  --cands_tsv  <out>/cands.tsv from 10_gen_candidates.py (columns utt cand wav ...); references come
               from --manifest: seame_normalize(text_raw) recomputed HERE (same env as the hypothesis;
               the manifest's text_ref column is only a fallback when text_raw is empty).
  --wav_scp    a Kaldi dir (or its wav.scp file): scores wav.scp against the dir's `text` with
               cand='gt' (override with --cand_label), so real and synthesized sets share one tool.
Duplicate (utt,cand) rows in cands.tsv are collapsed (last occurrence wins).

Output TSV columns: utt cand wav hyp ref n_ref edits mer   (hyp/ref are the normalised strings).
Corpus MER = sum(edits) / sum(n_ref) over rows with n_ref > 0 is printed and logged.
Rows whose wav cannot be loaded are skipped (not scored), listed in <out>.failed.txt (rewritten on
every run: `utt cand wav reason`) and reported as `failed N` in the summary line; re-run with
--resume to retry exactly those rows. --resume first repairs an --out whose last line was cut
short by a hard kill (common.repair_torn_tsv: the partial line is truncated away and its row
scored again) so nothing is glued onto a torn tail.
Exit status (shared critic policy, same flags in 12_/13_): 0 when the failed fraction
(failed / rows in scope) is <= --max_fail_frac (default 0.01; a WARNING is logged and the
afterok chain / .done marker proceed); 2 when it is above that fraction, or with --strict and
any failure at all.

Paths: --model_dir defaults to the configured CMI_DPO_ASR_MODEL_DIR (cmi_dpo.paths, config/paths.env
or the environment), resolved after parsing; --show_paths prints the configured paths and exits.

Example:
  python -u scripts/11_score_mer.py --cands_tsv exp/round1/cands.tsv \
      --manifest data/manifest_train.tsv --out exp/round1/mer.tsv --batch_size 16 --jiwer_check
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Iterator

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmi_dpo import common, paths  # noqa: E402

COLUMNS = ["utt", "cand", "wav", "hyp", "ref", "n_ref", "edits", "mer"]
LOG = logging.getLogger("score_mer")


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
        description="Score MER of candidate or real wavs with the SEAME-fine-tuned Whisper (env asr-whisper).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cands_tsv", default=None,
                     help="cands.tsv from 10_gen_candidates.py (columns utt cand wav ...).")
    src.add_argument("--wav_scp", default=None,
                     help="Kaldi data dir (or path to its wav.scp); scores wav.scp vs `text` with cand=--cand_label.")
    p.add_argument("--manifest", default=None,
                   help="Manifest TSV (00_prep_manifest.py) giving text_ref per utt; required with --cands_tsv, "
                        "optional override of the Kaldi `text` with --wav_scp.")
    p.add_argument("--out", required=True, help="Output TSV path (columns: utt cand wav hyp ref n_ref edits mer).")
    p.add_argument("--model_dir", default=None,
                   help="HF Whisper model dir used as the MER critic (default: the configured CMI_DPO_ASR_MODEL_DIR).")
    p.add_argument("--language", default=None,
                   help="Force Whisper decoding language (e.g. 'english'); default = model's own behaviour.")
    p.add_argument("--batch_size", type=int, default=8, help="Pipeline batch size (utterances per generate call).")
    p.add_argument("--limit", type=int, default=0, help="Only score the first N rows (smoke tests); 0 = all.")
    p.add_argument("--cand_label", default="gt", help="cand value written for --wav_scp rows.")
    p.add_argument("--resume", action="store_true",
                   help="Append to an existing --out, skipping (utt,cand) rows already scored "
                        "(a torn last line is repaired first).")
    p.add_argument("--jiwer_check", action="store_true",
                   help="Also compute corpus WER with jiwer on the normalised strings and assert agreement within 1e-6.")
    p.add_argument("--max_fail_frac", type=float, default=0.01,
                   help="Exit 0 (with a WARNING) when failed rows / rows in scope <= this fraction; exit 2 above it.")
    p.add_argument("--strict", action="store_true", help="Exit 2 on ANY failed row (ignores --max_fail_frac).")
    p.add_argument("--show_paths", action=ShowPathsAction)
    return p


def resolve_kaldi_dir(path: str) -> str:
    """Accept either a Kaldi dir or a path to its wav.scp."""
    if os.path.isfile(path):
        return os.path.dirname(os.path.abspath(path))
    return path


def dedupe_rows(rows: list[dict]) -> list[dict]:
    """Collapse duplicate (utt, cand) rows (append-only cands.tsv): last occurrence wins, first position kept."""
    uniq: dict[tuple[str, str], dict] = {}
    for r in rows:
        uniq[(r["utt"], r["cand"])] = r
    if len(uniq) != len(rows):
        LOG.warning("%d duplicate (utt,cand) rows dropped (last occurrence kept)", len(rows) - len(uniq))
    return list(uniq.values())


def load_rows(args: argparse.Namespace) -> list[dict]:
    """Return rows {utt, cand, wav} to decode, in file order (duplicate (utt,cand) keys collapsed)."""
    if args.cands_tsv:
        rows = [{"utt": r["utt"], "cand": r["cand"], "wav": r["wav"]} for r in common.read_tsv(args.cands_tsv)]
        LOG.info("read %d candidate rows from %s", len(rows), args.cands_tsv)
        rows = dedupe_rows(rows)
    else:
        d = resolve_kaldi_dir(args.wav_scp)
        kd = common.read_kaldi_dir(d)
        rows = [{"utt": u, "cand": args.cand_label, "wav": kd[u]["wav"]} for u in kd]
        LOG.info("read %d wav.scp rows from %s", len(rows), d)
    if args.limit > 0:
        rows = rows[: args.limit]
    return rows


def load_refs(args: argparse.Namespace, utts: list[str]) -> dict[str, str]:
    """Normalised reference per utt: manifest text_raw (else text_ref) wins over Kaldi text.

    The reference is ALWAYS re-normalised here, in the same env as the hypothesis (seame_normalize
    is env-dependent: opencc t2s only where importable; it is idempotent). The manifest's
    pre-normalised text_ref is advisory only (docs/DESIGN.md) and is used solely when text_raw is empty.
    """
    refs: dict[str, str] = {}
    if args.wav_scp:
        kd = common.read_kaldi_dir(resolve_kaldi_dir(args.wav_scp))
        for u, e in kd.items():
            refs[u] = common.seame_normalize(e["text_raw"])
    if args.manifest:
        n_from_ref = 0
        for r in common.read_manifest(args.manifest):
            raw = (r.get("text_raw") or "").strip()
            if not raw:
                raw = (r.get("text_ref") or "").strip()
                n_from_ref += 1
            refs[r["utt"]] = common.seame_normalize(raw)
        LOG.info("manifest %s: %d refs normalised in-process (%d fell back to text_ref)",
                 args.manifest, len(refs), n_from_ref)
    missing = [u for u in utts if u not in refs]
    if missing:
        raise SystemExit(f"{len(missing)} utts have no reference (first: {missing[0]}); pass --manifest")
    return refs


def read_done(path: str) -> set[tuple[str, str]]:
    """(utt, cand) keys already present in ``path`` (--resume); the header must match COLUMNS.

    A partial last line (interrupted append) is truncated away first (common.repair_torn_tsv)
    so the file is safe to append to and its row is decoded again.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
    if header != COLUMNS:
        raise ValueError(f"{path}: existing columns {header} differ from {COLUMNS}; run without --resume to rewrite it")
    removed = common.repair_torn_tsv(path, len(COLUMNS))
    if removed:
        LOG.warning("resume: repaired torn tail of %s (%d bytes truncated)", path, removed)
    done = {(r["utt"], r["cand"]) for r in common.read_tsv(path)}
    LOG.info("resume: %d rows already in %s", len(done), path)
    return done


def audio_inputs(rows: list[dict], failed: dict[int, str]) -> Iterator[dict]:
    """Generator consumed by the HF pipeline (same idiom as baselines/src/run_whisper.py).

    Per-row isolation: an unreadable/missing wav is recorded in ``failed[i]`` and replaced by a
    short silent array so the pipeline keeps its 1:1 input/output alignment; the caller must
    skip those indices instead of writing their (meaningless) hypotheses.
    """
    for i, r in enumerate(rows):
        try:
            arr = np.ascontiguousarray(common.load_audio16k(r["wav"]), dtype=np.float32)
            if arr.size == 0 or not np.isfinite(arr).all():
                raise ValueError("empty or non-finite audio")
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the whole decode
            failed[i] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
            LOG.error("utt=%s cand=%s: cannot load %s (%s); row skipped", r["utt"], r["cand"], r["wav"], failed[i])
            arr = np.zeros(8000, dtype=np.float32)  # 0.5 s placeholder, output discarded
        yield {"array": arr, "sampling_rate": 16000}


def build_asr(model_dir: str, batch_size: int):
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    LOG.info("loading ASR pipeline from %s (device=%s dtype=%s)", model_dir, device, dtype)
    return pipeline(
        "automatic-speech-recognition",
        model=model_dir,
        torch_dtype=dtype,
        device=device,
        chunk_length_s=30,
        batch_size=batch_size,
    )


def corpus_mer(rows: list[dict]) -> tuple[float, int, int]:
    edits = sum(int(r["edits"]) for r in rows if int(r["n_ref"]) > 0)
    n_ref = sum(int(r["n_ref"]) for r in rows if int(r["n_ref"]) > 0)
    return (edits / n_ref if n_ref else 0.0), edits, n_ref


def jiwer_check(rows: list[dict], mer: float) -> None:
    import jiwer

    scored = [r for r in rows if int(r["n_ref"]) > 0]
    wer = jiwer.wer([r["ref"] for r in scored], [r["hyp"] for r in scored])
    LOG.info("jiwer corpus WER = %.6f vs levenshtein corpus MER = %.6f", wer, mer)
    assert abs(wer - mer) < 1e-6, f"jiwer WER {wer} != corpus MER {mer}"


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cands_tsv and not args.manifest:
        raise SystemExit("--cands_tsv requires --manifest for the references")
    if not 0.0 <= args.max_fail_frac <= 1.0:
        raise SystemExit("--max_fail_frac must be in [0, 1]")
    if args.model_dir is None:  # resolved lazily so that --help works without config/paths.env
        args.model_dir = paths.asr_model_dir()

    rows = load_rows(args)
    refs = load_refs(args, [r["utt"] for r in rows])
    done = read_done(args.out) if args.resume else set()
    todo = [r for r in rows if (r["utt"], r["cand"]) not in done]
    LOG.info("%d rows to decode (%d skipped as done)", len(todo), len(rows) - len(todo))
    if not (args.resume and os.path.exists(args.out) and os.path.getsize(args.out) > 0):
        common.write_tsv(args.out, [], COLUMNS)  # header always exists, even with nothing to decode
    failed_txt = args.out + ".failed.txt"
    common.reset_failed(failed_txt)  # every row in scope is (re)attempted below, so the old list is stale

    failed: dict[int, str] = {}
    if todo:
        gen_kwargs = {"task": "transcribe"}
        if args.language:
            gen_kwargs["language"] = args.language
        asr = build_asr(args.model_dir, args.batch_size)
        outputs = asr(audio_inputs(todo, failed), batch_size=args.batch_size, generate_kwargs=gen_kwargs)
        for i, (r, out) in enumerate(zip(todo, outputs)):  # pipeline preserves input order
            if i in failed:
                common.append_failed(failed_txt, r["utt"], r["cand"], r["wav"], failed[i])
            else:
                hyp = common.seame_normalize(out["text"].strip())
                ref = refs[r["utt"]]
                edits, n_ref, mer = common.mer_pair(ref, hyp)
                common.append_tsv(args.out, {"utt": r["utt"], "cand": r["cand"], "wav": r["wav"], "hyp": hyp,
                                             "ref": ref, "n_ref": n_ref, "edits": edits, "mer": f"{mer:.6f}"},
                                  COLUMNS)
            if (i + 1) % 500 == 0 or i + 1 == len(todo):
                LOG.info("decoded %d/%d (%d failed)", i + 1, len(todo), len(failed))

    scored = common.read_tsv(args.out)
    mer, edits, n_ref = corpus_mer(scored)
    n_empty = sum(1 for r in scored if int(r["n_ref"]) == 0)
    LOG.info("corpus MER = %.4f %% (%d edits / %d ref tokens, %d rows, %d empty-ref rows excluded)",
             100.0 * mer, edits, n_ref, len(scored), n_empty)
    if args.jiwer_check:
        jiwer_check(scored, mer)
    print(f"corpus_MER% {100.0 * mer:.2f} rows {len(scored)} edits {edits} n_ref {n_ref} failed {len(failed)} "
          f"out {args.out}")
    return common.failure_exit_code(len(failed), len(rows), args.max_fail_frac, args.strict, failed_txt, LOG)


if __name__ == "__main__":
    sys.exit(main())
