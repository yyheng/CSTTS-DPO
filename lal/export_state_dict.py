"""Convert a whole-pickled ``WhisperWithLAL`` checkpoint into a portable state-dict file.

Purpose
    ``train_whisper_LAL.py`` saves checkpoints with ``torch.save(model)`` (a pickle of the
    module object), which can only be unpickled in an environment whose torch / transformers
    versions match the training one. This tool loads such a pickle ONCE (in that environment)
    and writes a plain state dict plus the metadata needed to rebuild the module anywhere:

        {'state_dict': ..., 'base_model': str, 'layer_index': int, 'n_lang': int,
         'd_model': int, 'format': 'cmi_dpo_lal_state_dict_v1'}

    ``cmi_dpo.lal_cmi.load_lal_model`` accepts either file. A ``<out>.json`` sidecar with the
    same metadata (without the tensors) is written next to the output for inspection.

Environment
    The env that produced the pickle (the Whisper-LAL training env). CPU only (no GPU needed).

Example
    python lal/export_state_dict.py --ckpt exp/loss_0.201_step_110000_wer_0.6204.pt \\
        --out exp/loss_0.201_step_110000_wer_0.6204.state_dict.pt
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys

import torch

STATE_DICT_FORMAT = "cmi_dpo_lal_state_dict_v1"
_DEBUG_ENV_KEYS: tuple[str, ...] = ("CUDA_LAUNCH_BLOCKING", "TORCH_SHOW_CPP_STACKTRACES")
LOG = logging.getLogger("export_state_dict")


def import_whisperlal() -> None:
    """Import ``WhisperLAL`` from this file's directory and undo its import-time debug side effects."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    saved = {k: os.environ.get(k) for k in _DEBUG_ENV_KEYS}
    importlib.import_module("WhisperLAL")
    torch.autograd.set_detect_anomaly(False)
    for key, old in saved.items():
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="whole-pickled WhisperWithLAL checkpoint (torch.save(model))")
    p.add_argument("--out", required=True, help="output state-dict file (.pt)")
    p.add_argument("--base_model", default="openai/whisper-small",
                   help="HF id / local dir of the base Whisper model the pickle was built from; stored as "
                        "'base_model' and used by load_lal_model to rebuild the module (default openai/whisper-small)")
    return p


def export(ckpt: str, out: str, base_model: str) -> int:
    """Load the pickle on CPU, write the state-dict file (+ json sidecar); returns the tensor count."""
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)
    import_whisperlal()
    try:
        model = torch.load(ckpt, map_location="cpu", weights_only=False)  # whole module: not weights-only
    except TypeError:  # torch without the weights_only kwarg
        model = torch.load(ckpt, map_location="cpu")
    if not hasattr(model, "whisper") or not hasattr(model, "language_cls"):
        raise TypeError(f"{ckpt} did not unpickle to a WhisperWithLAL model (got {type(model).__name__})")
    state_dict = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    payload = {
        "state_dict": state_dict,
        "base_model": base_model,
        "layer_index": int(getattr(model, "layer_index", -1)),
        "n_lang": int(model.language_cls.out_features),
        "d_model": int(model.whisper.config.d_model),
        "format": STATE_DICT_FORMAT,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    tmp = out + ".partial"
    torch.save(payload, tmp)
    os.replace(tmp, out)
    meta = {k: v for k, v in payload.items() if k != "state_dict"}
    meta.update({"source_ckpt": os.path.abspath(ckpt), "n_tensors": len(state_dict),
                 "n_params": int(sum(v.numel() for v in state_dict.values()))})
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    LOG.info("wrote %s (%d tensors, %.1fM params, base_model=%s layer_index=%d n_lang=%d d_model=%d)",
             out, len(state_dict), meta["n_params"] / 1e6, base_model, payload["layer_index"],
             payload["n_lang"], payload["d_model"])
    return len(state_dict)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    n = export(args.ckpt, args.out, args.base_model)
    print(f"EXPORT_OK tensors={n} out={args.out}", flush=True)


if __name__ == "__main__":
    main()
