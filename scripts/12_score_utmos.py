"""UTMOS critic: predict naturalness MOS (UTMOS22 strong learner) for candidate or real wavs.

Env: GPU; pure torch + torchaudio (the CosyVoice env is enough).

Model (``--utmos_source``, default = the configured CMI_DPO_UTMOS_SOURCE, see cmi_dpo.paths):
  hub    ``torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True)``: the
         repo and the checkpoint are downloaded ONCE into the torch.hub cache (needs internet the
         first time; honours ``TORCH_HOME``, default ~/.cache/torch) and reused offline afterwards.
  local  ``speechmos_ut.utmos22.strong.model.UTMOS22Strong`` imported from a local clone of the
         SpeechMOS repo (--utmos_repo, default CMI_DPO_UTMOS_REPO; the same class hubconf.py's
         ``utmos22_strong`` builds) with the state_dict torch.hub would download
         (--utmos_ckpt, default CMI_DPO_UTMOS_CKPT, e.g. utmos22_strong_step7459_v1.pt). No network.
Both give the same module: ``forward(wave [B,T] float32, sr) -> [B]`` resamples internally to
16 kHz, so each file is scored at its own sample rate, one utterance at a time under
torch.no_grad(). The defaults are resolved after parsing, so --help works without the CMI_DPO_*
variables; --show_paths prints the configured paths and exits.

Input modes (exactly one): --cands_tsv (utt cand wav ...) or --wav_scp (Kaldi dir or wav.scp,
cand=--cand_label, default 'gt'). Duplicate (utt,cand) rows in cands.tsv are collapsed (last wins).
Output TSV columns: utt cand wav utmos

Per-row failures are isolated (same approach as 11_score_mer.py): a missing/unreadable wav, an
empty or non-finite waveform, a model exception or a non-finite MOS is logged, listed in
<out>.failed.txt (rewritten on every run: `utt cand wav reason`), reported as `failed N` in the
summary line, and scoring continues; re-run with --resume to retry exactly those rows. --resume
first repairs an --out whose last line was cut short by a hard kill (common.repair_torn_tsv:
the partial line is truncated away and its row scored again) so nothing is glued onto a torn tail.
Exit status (shared critic policy, same flags in 11_/13_): 0 when the failed fraction
(failed / rows in scope) is <= --max_fail_frac (default 0.01; a WARNING is logged and
downstream steps that require exit status 0 proceed); 2 when it is above that fraction, or
with --strict and any failure at all.

Example:
  python -u scripts/12_score_utmos.py --cands_tsv exp/round1/cands.tsv --out exp/round1/utmos.tsv
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmi_dpo import common, paths  # noqa: E402

HUB_REPO = "tarepan/SpeechMOS:v1.2.0"
HUB_MODEL = "utmos22_strong"
UTMOS_SOURCES = ("hub", "local")
COLUMNS = ["utt", "cand", "wav", "utmos"]
LOG = logging.getLogger("score_utmos")


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
        description="Score UTMOS22-strong naturalness MOS for candidate or real wavs (GPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cands_tsv", default=None, help="cands.tsv from 10_gen_candidates.py (utt cand wav ...).")
    src.add_argument("--wav_scp", default=None, help="Kaldi data dir (or path to its wav.scp); cand=--cand_label.")
    p.add_argument("--out", required=True, help="Output TSV path (columns: utt cand wav utmos).")
    p.add_argument("--utmos_source", choices=UTMOS_SOURCES, default=None,
                   help=f"hub = torch.hub.load('{HUB_REPO}', '{HUB_MODEL}') (downloads once, honours TORCH_HOME); "
                        "local = --utmos_repo + --utmos_ckpt (default: the configured CMI_DPO_UTMOS_SOURCE).")
    p.add_argument("--utmos_repo", "--repo", dest="utmos_repo", default=None,
                   help="local: SpeechMOS repo clone providing speechmos_ut (default: CMI_DPO_UTMOS_REPO).")
    p.add_argument("--utmos_ckpt", "--ckpt", dest="utmos_ckpt", default=None,
                   help="local: UTMOS22 strong state_dict checkpoint (default: CMI_DPO_UTMOS_CKPT).")
    p.add_argument("--device", default="cuda:0", help="torch device.")
    p.add_argument("--limit", type=int, default=0, help="Only score the first N rows; 0 = all.")
    p.add_argument("--cand_label", default="gt", help="cand value written for --wav_scp rows.")
    p.add_argument("--resume", action="store_true",
                   help="Append to an existing --out, skipping (utt,cand) rows already scored "
                        "(a torn last line is repaired first).")
    p.add_argument("--max_fail_frac", type=float, default=0.01,
                   help="Exit 0 (with a WARNING) when failed rows / rows in scope <= this fraction; exit 2 above it.")
    p.add_argument("--strict", action="store_true", help="Exit 2 on ANY failed row (ignores --max_fail_frac).")
    p.add_argument("--show_paths", action=ShowPathsAction)
    return p


def resolve_utmos_args(args: argparse.Namespace) -> None:
    """Fill --utmos_source / --utmos_repo / --utmos_ckpt from cmi_dpo.paths when not given (in place).

    Done after parsing so that --help works without the CMI_DPO_* variables. 'local' needs both a
    repo and a checkpoint (SystemExit otherwise); 'hub' ignores them.
    """
    if args.utmos_source is None:
        args.utmos_source = paths.utmos_source()
    if args.utmos_source not in UTMOS_SOURCES:
        raise SystemExit(f"CMI_DPO_UTMOS_SOURCE / --utmos_source must be one of {UTMOS_SOURCES}, got {args.utmos_source!r}")
    if args.utmos_source == "local":
        if args.utmos_repo is None:
            args.utmos_repo = paths.utmos_repo()
        if args.utmos_ckpt is None:
            args.utmos_ckpt = paths.utmos_ckpt()
        for flag, var, value in (("--utmos_repo", "CMI_DPO_UTMOS_REPO", args.utmos_repo),
                                 ("--utmos_ckpt", "CMI_DPO_UTMOS_CKPT", args.utmos_ckpt)):
            if not value:
                raise SystemExit(f"--utmos_source local needs {flag} (or export {var}); "
                                 f"use --utmos_source hub to download the model with torch.hub instead")


def resolve_kaldi_dir(path: str) -> str:
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
    """Rows {utt, cand, wav} to score, in file order (duplicate (utt,cand) keys collapsed)."""
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


def load_utmos_local(repo: str, ckpt: str, device: torch.device) -> torch.nn.Module:
    """UTMOS22Strong from a local SpeechMOS clone + state_dict (no network)."""
    sys.path.insert(0, repo)
    from speechmos_ut.utmos22.strong.model import UTMOS22Strong

    model = UTMOS22Strong()
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)  # strict: keys must match exactly
    model.eval().to(device)
    LOG.info("loaded UTMOS22Strong from %s (%d tensors) on %s", ckpt, len(state), device)
    return model


def load_utmos_hub(device: torch.device) -> torch.nn.Module:
    """UTMOS22Strong via torch.hub (downloads the repo + checkpoint once into $TORCH_HOME/hub)."""
    LOG.info("loading %s '%s' with torch.hub (cache: %s)", HUB_REPO, HUB_MODEL, torch.hub.get_dir())
    model = torch.hub.load(HUB_REPO, HUB_MODEL, trust_repo=True)
    model.eval().to(device)
    LOG.info("loaded %s from torch.hub on %s", type(model).__name__, device)
    return model


def load_utmos(source: str, repo: Optional[str], ckpt: Optional[str], device: torch.device) -> torch.nn.Module:
    """Dispatch on --utmos_source ('hub' | 'local'); see the module docstring."""
    if source == "hub":
        return load_utmos_hub(device)
    if source == "local":
        return load_utmos_local(repo or "", ckpt or "", device)
    raise ValueError(f"unknown utmos source {source!r}")


def load_wave(path: str) -> tuple[torch.Tensor, int]:
    """Read any-sample-rate audio as mono float32 [1,T] plus its native sr.

    Raises ValueError for an empty or non-finite waveform (a diverged vocoder output would
    otherwise be scored silently); unreadable/missing files raise from soundfile.
    """
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("empty audio")
    if not np.isfinite(audio).all():
        raise ValueError("non-finite samples")
    return torch.from_numpy(audio).unsqueeze(0), int(sr)


@torch.no_grad()
def score_wave(model: torch.nn.Module, wave: torch.Tensor, sr: int, device: torch.device) -> float:
    mos = float(model(wave.to(device), sr)[0].item())
    if not np.isfinite(mos):
        raise ValueError(f"non-finite MOS {mos}")
    return mos


def score_row(model: torch.nn.Module, r: dict, device: torch.device) -> tuple[Optional[float], Optional[str]]:
    """Score one row; returns (mos, None) or (None, reason) so a bad file never aborts the run."""
    try:
        wave, sr = load_wave(r["wav"])
        return score_wave(model, wave, sr, device), None
    except Exception as exc:  # noqa: BLE001 - unreadable/missing/non-finite file or model error: report, keep going
        first = str(exc).splitlines()[0] if str(exc) else ""
        return None, f"{type(exc).__name__}: {first}"


def read_done(path: str) -> set[tuple[str, str]]:
    """(utt, cand) keys already present in ``path`` (--resume); the header must match COLUMNS.

    A partial last line (interrupted append) is truncated away first (common.repair_torn_tsv)
    so the file is safe to append to and its row is scored again.
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


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not 0.0 <= args.max_fail_frac <= 1.0:
        raise SystemExit("--max_fail_frac must be in [0, 1]")
    resolve_utmos_args(args)

    rows = load_rows(args)
    done = read_done(args.out) if args.resume else set()
    if not (args.resume and os.path.exists(args.out) and os.path.getsize(args.out) > 0):
        common.write_tsv(args.out, [], COLUMNS)  # header always exists, even with nothing to score
    todo = [r for r in rows if (r["utt"], r["cand"]) not in done]
    LOG.info("%d rows to score (%d skipped as done)", len(todo), len(rows) - len(todo))
    failed_txt = args.out + ".failed.txt"
    common.reset_failed(failed_txt)  # every row in scope is (re)attempted below, so the old list is stale

    n_failed = 0
    if todo:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model = load_utmos(args.utmos_source, args.utmos_repo, args.utmos_ckpt, device)
        for i, r in enumerate(todo, 1):
            mos, why = score_row(model, r, device)
            if mos is None:
                n_failed += 1
                LOG.error("FAILED utt=%s cand=%s wav=%s: %s; row skipped", r["utt"], r["cand"], r["wav"], why)
                common.append_failed(failed_txt, r["utt"], r["cand"], r["wav"], why or "")
            else:
                common.append_tsv(args.out, {"utt": r["utt"], "cand": r["cand"], "wav": r["wav"], "utmos": f"{mos:.4f}"},
                                  COLUMNS)
            if i % 500 == 0 or i == len(todo):
                LOG.info("scored %d/%d (%d failed)", i, len(todo), n_failed)

    scored = common.read_tsv(args.out)
    vals = [float(r["utmos"]) for r in scored]
    mean = float(np.mean(vals)) if vals else 0.0
    LOG.info("mean UTMOS = %.4f over %d rows (%d failed)", mean, len(vals), n_failed)
    print(f"mean_UTMOS {mean:.4f} rows {len(vals)} failed {n_failed} out {args.out}")
    return common.failure_exit_code(n_failed, len(rows), args.max_fail_frac, args.strict, failed_txt, LOG)


if __name__ == "__main__":
    sys.exit(main())
