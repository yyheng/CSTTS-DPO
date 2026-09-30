#!/usr/bin/env python
"""Build the stage-1 SFT token cache for CosyVoice2 LLM fine-tuning (see docs/DESIGN.md).

For every row of a manifest TSV (columns: utt spk dur wav text_raw text_tts
text_ref prompt_utt prompt_wav prompt_text) this script
  * tokenises ``text_tts`` with ``cv.frontend._extract_text_token`` (Qwen tokenizer, no
    ttsfrd normalisation unless ``--text_frontend`` is given),
  * extracts 25 tok/s speech tokens with ``cv.frontend._extract_speech_token`` from the
    16 kHz waveform loaded by ``cosyvoice.utils.file_utils.load_wav``,
  * skips utterances whose audio exceeds ``--max_dur`` seconds (speech tokenizer limit 30 s)
    or whose text token count is < ``--min_text_tokens`` or > ``--max_text_tokens``,
and saves a list of ``{'utt', 'text_token' int32 cpu [L], 'speech_token' int32 cpu [T]}``
to ``--out`` (a .pt file) plus ``<out stem>.json`` with the kept/skipped counts.

The manifest can be split across jobs with ``--shard i --nshards n`` (one .pt per shard);
``02_train_sft.py`` accepts several shards.

Resumable: every ``--log_every`` rows the partial result is written to ``<out>.partial``;
a rerun with the same arguments loads it and continues from the first unprocessed row
(``--overwrite`` discards it). The partial file is deleted once ``--out`` is written.

``--text_col`` picks the manifest column that is tokenised (default ``text_tts``; use
``text_raw`` to mirror the authors' epoch_020.pth convention with <noise>/<UNK> tags kept
when the manifest was built with ``--text_mode strip``, see docs/DESIGN.md).

Environment: the CosyVoice env (torch 2.3.1, onnxruntime, openai-whisper).
CPU is enough (GPU optional; the speech tokenizer is ONNX, the Qwen tokenizer is CPU).

Paths: the CosyVoice2 checkout / model dir come from cmi_dpo.paths (CMI_DPO_COSY_ROOT,
CMI_DPO_COSY_MODEL_DIR in the environment); --show_paths prints them and exits.

Example (one shard of four):
    python -u scripts/01_build_sft_cache.py \\
        --manifest data/train.tsv \\
        --out data/sft_cache/train.shard0.pt \\
        --shard 0 --nshards 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmi_dpo import common  # noqa: E402
from cmi_dpo import cosy  # noqa: E402
from cmi_dpo import paths  # noqa: E402

LOG = logging.getLogger("build_sft_cache")


class ShowPathsAction(argparse.Action):
    """--show_paths: print cmi_dpo.paths.describe() and exit, before the required flags are checked."""

    def __init__(self, option_strings: list[str], dest: str, **kwargs: Any) -> None:
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS,
                         help=kwargs.get("help", "print the configured paths (cmi_dpo.paths) and exit"))

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
                 values: Any, option_string: str | None = None) -> None:
        print(paths.describe())
        parser.exit()

SKIP_REASONS = ("too_long", "empty_audio", "text_too_short", "text_too_long", "load_error")
# cosyvoice/cli/frontend.py _extract_speech_token asserts speech.shape[1] / 16000 <= 30.
MAX_SPEECH_TOKENIZER_S = 30.0
# Arguments that must match between a run and the <out>.partial file it resumes from.
_RESUME_KEYS = ("manifest", "shard", "nshards", "limit", "text_frontend", "text_col",
                "max_dur", "min_text_tokens", "max_text_tokens")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", required=True,
                   help="Manifest TSV from 00_prep_manifest.py (needs columns utt, wav, text_tts).")
    p.add_argument("--out", required=True,
                   help="Output cache .pt (list of dicts). Counts are written to <out stem>.json.")
    p.add_argument("--shard", type=int, default=0,
                   help="Index of this shard in [0, nshards).")
    p.add_argument("--nshards", type=int, default=1,
                   help="Number of shards the manifest is split into (round-robin, see common.shard).")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N rows of this shard (0 = all). For smoke tests.")
    p.add_argument("--device", default="cuda:0",
                   help="Device for CosyVoice2 (frontend tokenisers run on ONNX/CPU regardless).")
    p.add_argument("--text_frontend", action="store_true",
                   help="Apply frontend.text_normalize(text, split=False, text_frontend=True) "
                        "(ttsfrd) before tokenising. Default: tokenise text_tts as-is.")
    p.add_argument("--text_col", choices=("text_tts", "text_raw"), default="text_tts",
                   help="Manifest column to tokenise (text_raw keeps <noise>/<UNK> tags even when "
                        "the manifest was built with --text_mode strip).")
    p.add_argument("--max_dur", type=float, default=30.0,
                   help="Skip utterances whose loaded audio is longer than this many seconds "
                        f"(clamped to {MAX_SPEECH_TOKENIZER_S:.0f}: the speech tokenizer asserts <= 30 s).")
    p.add_argument("--min_text_tokens", type=int, default=1,
                   help="Skip utterances with fewer text tokens than this.")
    p.add_argument("--max_text_tokens", type=int, default=200,
                   help="Skip utterances with more text tokens than this.")
    p.add_argument("--log_every", type=int, default=500,
                   help="Log progress every N processed rows.")
    p.add_argument("--overwrite", action="store_true",
                   help="Rebuild even if --out already exists (default: refuse to clobber).")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed (extraction is deterministic; kept for uniform CLI).")
    p.add_argument("--show_paths", action=ShowPathsAction)
    return p.parse_args(argv)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )


def text_tokens_for(cv: Any, text_tts: str, text_frontend: bool) -> torch.Tensor:
    """Return the int32 cpu [L] text token vector for one utterance."""
    text = text_tts
    if text_frontend:
        text = cv.frontend.text_normalize(text, split=False, text_frontend=True)
    text_token, _ = cv.frontend._extract_text_token(text)
    return text_token.squeeze(0).to(torch.int32).cpu()


def speech_tokens_for(cv: Any, speech16k: torch.Tensor) -> torch.Tensor:
    """Return the int32 cpu [T] speech token vector for a [1, T] 16 kHz waveform."""
    speech_token, _ = cv.frontend._extract_speech_token(speech16k)
    return speech_token.squeeze(0).to(torch.int32).cpu()


def process_row(cv: Any, row: dict, load_wav: Any, args: argparse.Namespace) -> tuple[dict | None, str | None]:
    """Build one cache entry. Returns (entry, None) on success or (None, skip_reason)."""
    text_token = text_tokens_for(cv, row[args.text_col], args.text_frontend)
    n_text = int(text_token.shape[0])
    if n_text < args.min_text_tokens:
        return None, "text_too_short"
    if n_text > args.max_text_tokens:
        return None, "text_too_long"

    speech = load_wav(row["wav"], 16000)
    n_samples = int(speech.shape[1])
    if n_samples < 160:  # < 1 frame at 100 fps
        return None, "empty_audio"
    if n_samples / 16000.0 > args.max_dur:
        return None, "too_long"

    speech_token = speech_tokens_for(cv, speech)
    if int(speech_token.shape[0]) < 1:
        return None, "empty_audio"
    return {"utt": row["utt"], "text_token": text_token, "speech_token": speech_token}, None


def build_cache(args: argparse.Namespace) -> dict:
    """Run the extraction for this shard and write the .pt and .json outputs."""
    if not (0 <= args.shard < args.nshards):
        raise ValueError(f"--shard {args.shard} must be in [0, {args.nshards})")
    if os.path.exists(args.out) and not args.overwrite:
        raise FileExistsError(f"{args.out} exists; pass --overwrite to rebuild")
    if args.max_dur > MAX_SPEECH_TOKENIZER_S:
        LOG.warning("--max_dur %.1f > %.0f s: clamped (speech tokenizer asserts <= 30 s audio)",
                    args.max_dur, MAX_SPEECH_TOKENIZER_S)
        args.max_dur = MAX_SPEECH_TOKENIZER_S

    common.seed_all(args.seed)
    rows_all = common.read_manifest(args.manifest)
    rows = common.shard(rows_all, args.shard, args.nshards)
    if args.limit > 0:
        rows = rows[: args.limit]
    LOG.info("manifest %s: %d rows, shard %d/%d -> %d rows%s",
             args.manifest, len(rows_all), args.shard, args.nshards, len(rows),
             f" (limit {args.limit})" if args.limit > 0 else "")

    # resume from <out>.partial (written every --log_every rows) unless --overwrite
    partial_path = args.out + ".partial"
    entries: list[dict] = []
    skipped: dict[str, int] = {r: 0 for r in SKIP_REASONS}
    n_done = 0
    total_text = 0
    total_speech = 0
    elapsed_before = 0.0
    if os.path.exists(partial_path):
        if args.overwrite:
            LOG.info("--overwrite: discarding %s", partial_path)
            os.remove(partial_path)
        else:
            state = torch.load(partial_path, map_location="cpu")
            want = {k: getattr(args, k) for k in _RESUME_KEYS}
            want["n_shard_rows"] = len(rows)
            got = dict(state.get("args", {}))
            got["n_shard_rows"] = state.get("n_shard_rows")
            if got != want:
                raise ValueError(f"{partial_path} was written with different arguments "
                                 f"({got} != {want}); pass --overwrite to discard it")
            entries = state["entries"]
            skipped = {r: int(state["skipped"].get(r, 0)) for r in SKIP_REASONS}
            n_done = int(state["n_done"])
            total_text = int(state["total_text_tokens"])
            total_speech = int(state["total_speech_tokens"])
            elapsed_before = float(state.get("elapsed_s", 0.0))
            LOG.info("resuming from %s: %d/%d rows done, kept %d, skipped %s",
                     partial_path, n_done, len(rows), len(entries), skipped)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    def save_partial(i: int) -> None:
        state = {
            "args": {k: getattr(args, k) for k in _RESUME_KEYS},
            "n_shard_rows": len(rows),
            "n_done": i,
            "entries": entries,
            "skipped": skipped,
            "total_text_tokens": total_text,
            "total_speech_tokens": total_speech,
            "elapsed_s": elapsed_before + time.time() - t0,
        }
        tmp = partial_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, partial_path)

    device = torch.device(args.device)
    LOG.info("loading CosyVoice2 from %s on %s", paths.cosy_model_dir(), device)
    cv = cosy.load_cosyvoice2(device, llm_ckpt=None)
    cosy_root = paths.cosy_root()
    for p in (cosy_root, os.path.join(cosy_root, "Matcha-TTS")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from cosyvoice.utils.file_utils import load_wav  # noqa: E402  (needs COSY_ROOT on sys.path)

    t0 = time.time()
    for i, row in enumerate(rows[n_done:], n_done + 1):
        try:
            entry, reason = process_row(cv, row, load_wav, args)
        except Exception as e:  # corrupt audio, tokenizer failure: count, log, continue
            LOG.warning("utt %s: %s: %s", row.get("utt", "?"), type(e).__name__, e)
            entry, reason = None, "load_error"
        if entry is None:
            skipped[reason] += 1
        else:
            entries.append(entry)
            total_text += int(entry["text_token"].shape[0])
            total_speech += int(entry["speech_token"].shape[0])
        if i % args.log_every == 0 or i == len(rows):
            el = time.time() - t0
            LOG.info("%d/%d rows, kept %d, skipped %s, %.1f rows/s",
                     i, len(rows), len(entries), skipped, (i - n_done) / max(el, 1e-6))
            if i < len(rows):
                save_partial(i)

    torch.save(entries, args.out)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    counts = {
        "manifest": args.manifest,
        "out": args.out,
        "shard": args.shard,
        "nshards": args.nshards,
        "limit": args.limit,
        "text_frontend": args.text_frontend,
        "text_col": args.text_col,
        "max_dur": args.max_dur,
        "n_manifest": len(rows_all),
        "n_shard_rows": len(rows),
        "n_kept": len(entries),
        "n_skipped": int(sum(skipped.values())),
        "skipped": skipped,
        "total_text_tokens": total_text,
        "total_speech_tokens": total_speech,
        "speech_hours": total_speech / 25.0 / 3600.0,
        "elapsed_s": elapsed_before + time.time() - t0,
    }
    json_path = os.path.splitext(args.out)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2, ensure_ascii=False)
    LOG.info("wrote %s (%d entries, %.2f h of speech tokens) and %s",
             args.out, len(entries), counts["speech_hours"], json_path)
    return counts


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    build_cache(args)


if __name__ == "__main__":
    main()
