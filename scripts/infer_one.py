#!/usr/bin/env python
"""Synthesise ONE utterance with the (fine-tuned) CosyVoice2 model: the minimal inference entry point.

Purpose
    Zero-shot voice-cloned synthesis of a code-switching sentence from a 16 kHz speaker prompt,
    with exactly the conditioning path the pipeline uses (cmi_dpo.cosy: build_prompt_inputs ->
    generate_speech_tokens_ex -> tokens_to_wav): no ttsfrd text normalisation unless
    --text_frontend is given, RAS sampling with a temperature knob, flow + HiFT vocoding.

Text convention
    --text and --prompt_text follow the SEAME transcripts the models were trained on: Chinese
    characters separated by spaces, lower-case English words, e.g.
    'oh wednesday 是 我 们 的 lab 才 ah'. Any other text works too, but --text_frontend then
    applies CosyVoice's own normaliser (frontend.text_normalize, ttsfrd / wetext) to BOTH texts
    first, which rewrites numbers and punctuation and may merge the spaced characters.

Outputs
    --n 1 (default): --out is written as given (24 kHz mono float32 wav; --sr 16000 resamples
    with torchaudio.functional.resample before saving).
    --n K > 1: <out stem>_0.wav ... <out stem>_{K-1}.wav, one independent sample each (sample k
    re-seeds all RNGs from --seed + k). For every file one line
        <path>  dur=<seconds>  tokens=<n speech tokens>  eos=<True|False>
    is printed, then INFER_ONE_DONE. A sample cut by the max-length rule without an EOS
    (eos=False, runaway generation) is still written but flagged.

Model
    --ckpt: Qwen2LM state_dict (stage-1 SFT or DPO checkpoint, same loader); default = the
    configured CMI_DPO_SFT_CKPT (cmi_dpo.paths, config/paths.env), 'none' = the stock CosyVoice2
    llm.pt. CosyVoice code / model dirs come from CMI_DPO_COSY_ROOT / CMI_DPO_COSY_MODEL_DIR.
    --show_paths prints the configured paths and exits. Scoring the output (MER / UTMOS /
    CMIspeech) is NOT done here: see 11_score_mer.py, 12_score_utmos.py and 13_score_cmi.py.

Environment
    The CosyVoice env (CMI_DPO_ENV_MAIN; on the cluster ``cosyvoicenew``). GPU (--device cuda:0,
    default) or CPU (--device cpu, ~1 min per short utterance); on the cluster run it through
    srun/sbatch, never on the login node.

Example
    python -u scripts/infer_one.py \\
        --text 'oh wednesday 是 我 们 的 lab 才 ah' \\
        --prompt_wav $CMI_DPO_DATA_ROOT/devman/data/format.1/nc12m-06nc12may_0101-000460-000718.flac \\
        --prompt_text 'okay 好 你 你 介 绍 先 啦' \\
        --out exp/infer/one.wav --temperature 1.0 --seed 0
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmi_dpo import common  # noqa: E402
from cmi_dpo import cosy  # noqa: E402
from cmi_dpo import paths  # noqa: E402

MODEL_SR = 24000
MAX_SPEECH_TOKEN_ID = 6561  # EOS id; the vocoder never sees it (same guard as 10_/20_)
SUPPORTED_SR = (24000, 16000)
LOG = logging.getLogger("infer_one")


class ShowPathsAction(argparse.Action):
    """--show_paths: print cmi_dpo.paths.describe() and exit, before the required flags are checked."""

    def __init__(self, option_strings: list[str], dest: str, **kwargs: Any) -> None:
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS,
                         help=kwargs.get("help", "print the configured paths (cmi_dpo.paths) and exit"))

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
                 values: Any, option_string: str | None = None) -> None:
        print(paths.describe())
        parser.exit()


def resolve_ckpt(arg: str | None) -> str | None:
    """--ckpt value -> LLM checkpoint path, or None for the stock llm.pt.

    Flag omitted (None) = the configured CMI_DPO_SFT_CKPT (None when it is unset / empty);
    'none' (case-insensitive) or '' = the stock CosyVoice2 llm.pt; anything else = that path.
    Resolved after parsing so that --help works without config/paths.env.
    """
    if arg is None:
        return paths.sft_ckpt() or None
    return None if arg.strip().lower() in ("", "none") else arg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthesise one utterance with CosyVoice2 (SFT / DPO checkpoint) from a speaker prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--text", required=True,
                   help="text to synthesise (SEAME convention: Chinese characters space-separated; any text works, "
                        "--text_frontend applies CosyVoice's normaliser)")
    p.add_argument("--prompt_wav", required=True, help="speaker prompt audio (any sample rate, <= 30 s; loaded at 16 kHz)")
    p.add_argument("--prompt_text", required=True, help="transcript of --prompt_wav (same text convention as --text)")
    p.add_argument("--out", required=True, help="output wav path (<stem>_<k>.wav per sample when --n > 1)")
    p.add_argument("--ckpt", default=None,
                   help="Qwen2LM state_dict (stage-1 SFT or DPO checkpoint); default = the configured "
                        "CMI_DPO_SFT_CKPT (stock llm.pt when unset); 'none' = stock llm.pt")
    p.add_argument("--temperature", type=float, default=1.0, help="softmax temperature on the LLM logits (> 0)")
    p.add_argument("--sampling", type=int, default=25, help="`sampling` argument of Qwen2LM.sampling_ids (RAS fallback k)")
    p.add_argument("--seed", type=int, default=0, help="seed applied after model loading; sample k uses seed + k")
    p.add_argument("--n", type=int, default=1, help="number of independent samples to write")
    p.add_argument("--sr", type=int, choices=SUPPORTED_SR, default=MODEL_SR,
                   help="output sample rate (the model runs at 24000; 16000 resamples with torchaudio)")
    p.add_argument("--text_frontend", action="store_true",
                   help="apply frontend.text_normalize(split=False, text_frontend=True) to both texts before tokenising")
    p.add_argument("--device", default="cuda:0", help="torch device (falls back to cpu with a warning when CUDA is absent)")
    p.add_argument("--show_paths", action=ShowPathsAction)
    args = p.parse_args(argv)
    if args.temperature <= 0:
        p.error(f"--temperature must be > 0 (got {args.temperature})")
    if args.n < 1:
        p.error("--n must be >= 1")
    if not args.text.strip():
        p.error("--text is empty")
    if not os.path.isfile(args.prompt_wav):
        p.error(f"--prompt_wav not found: {args.prompt_wav}")
    return args


def out_paths(out: str, n: int) -> list[str]:
    """[out] for n == 1, else <stem>_<k><ext> for k in range(n) (ext defaults to .wav)."""
    if n == 1:
        return [out]
    stem, ext = os.path.splitext(out)
    return [f"{stem}_{k}{ext or '.wav'}" for k in range(n)]


def synthesise(cv: Any, inputs: dict, temperature: float, sampling: int) -> tuple[torch.Tensor, int, bool]:
    """One sample: (wav [1,S] float32 cpu @ 24 kHz, n_tokens, ended_with_eos)."""
    tokens, ended_with_eos = cosy.generate_speech_tokens_ex(cv.model.llm, inputs, temperature=temperature,
                                                            sampling=sampling)
    tokens = [t for t in tokens if t < MAX_SPEECH_TOKEN_ID]
    if not tokens:
        raise RuntimeError("generation produced zero speech tokens")
    wav = cosy.tokens_to_wav(cv, tokens, inputs)
    return wav, len(tokens), bool(ended_with_eos)


def main(argv: list[str] | None = None) -> None:
    common.setup_logging()
    args = parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    ckpt = resolve_ckpt(args.ckpt)  # None = stock llm.pt
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        LOG.warning("CUDA not available: running on cpu")
        device = torch.device("cpu")
    outs = out_paths(args.out, args.n)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    cv = cosy.load_cosyvoice2(device, llm_ckpt=ckpt)
    cv.model.llm.eval()
    LOG.info("CosyVoice2 loaded on %s (llm ckpt=%s)", device, ckpt or "stock llm.pt")
    inputs = cosy.build_prompt_inputs(cv, args.text.strip(), args.prompt_text.strip(), args.prompt_wav,
                                      text_frontend=args.text_frontend)
    LOG.info("prompt %s: %d speech tokens; target text %d tokens (text_frontend=%s)", args.prompt_wav,
             int(inputs["prompt_speech"].numel()), int(inputs["text_len_target"]), args.text_frontend)

    for k, path in enumerate(outs):
        common.seed_all(args.seed + k)  # after loading: cosyvoice.yaml re-seeds the global RNGs to 1986
        wav, n_tokens, eos = synthesise(cv, inputs, args.temperature, args.sampling)
        if not eos:
            LOG.warning("sample %d hit the max-length rule without EOS (%d tokens): possible runaway generation",
                        k, n_tokens)
        if args.sr != MODEL_SR:
            wav = common.resample(wav, MODEL_SR, args.sr)
        dur = wav.shape[1] / args.sr
        torchaudio.save(path, wav, args.sr)
        print(f"{path}\tdur={dur:.3f}s\ttokens={n_tokens}\teos={eos}", flush=True)
    print("INFER_ONE_DONE", flush=True)


if __name__ == "__main__":
    main()
