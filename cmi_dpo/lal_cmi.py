"""Whisper-LAL component: pseudo frame-level language labels and CMIspeech.

Purpose
    Load a Whisper-LAL checkpoint (``WhisperWithLAL``: openai/whisper-small plus a 4-way
    per-frame language classifier ``language_cls``, trained by lal/train_whisper_LAL.py) either
    as the whole pickled model or as the portable state dict written by
    lal/export_state_dict.py, turn 16 kHz audio into Whisper 80-bin log-mel features, predict
    one language class per 20 ms encoder frame and compute

        CMIspeech(u) = (T(u) - max_k T_k(u)) / T(u),   k in L,

    where T_k(u) is the number of frames labelled k and T(u) = sum_{k in L} T_k(u).
    Default L = {zh, en}; blank/other frames are excluded from T(u). ``langs=(0,1,2,3)``
    counts every class. T(u) == 0 gives CMI 0. ΔCMI = |CMIspeech(synth) - CMIspeech(gt)|.

    Frame classes (WhisperDataPreLAL.py label map): 0 = zh (CJK), 1 = en, 2 = blank, 3 = other.
    Whisper-small: 3000 mel frames (10 ms) per 30 s window -> 1500 encoder frames (20 ms).

Environment
    A whole-pickled checkpoint only unpickles where the torch / transformers versions of the
    training env are importable (the Whisper-LAL training env: torch 1.13.1+cu117,
    transformers 4.38.0, openai-whisper 20250625). A state-dict checkpoint
    (lal/export_state_dict.py) loads in any env with torch, transformers and openai-whisper.
    Both need ``import WhisperLAL`` from ``CMI_DPO_LAL_CODE_DIR`` (default: the vendored
    ``<repo>/lal``), done by ``load_lal_model``. The pure helpers (``cmi_from_labels``,
    ``delta_cmi``, ``labels_rle``, ``n_valid_frames``) need only numpy and run in any env.

Example (GPU, LAL env, repo root on sys.path)
    python -c "import torch; from cmi_dpo import common, lal_cmi, paths; \\
        m = lal_cmi.load_lal_model(paths.lal_ckpt(), 'cuda:0'); \\
        a = common.load_audio16k(paths.data_root() + '/devman/data/format.1/nc12m-06nc12may_0101-000162-000460.flac'); \\
        lab = lal_cmi.frame_language_labels(m, lal_cmi.whisper_mel(a).unsqueeze(0), [len(a) / 16000])[0]; \\
        print(lal_cmi.cmi_from_labels(lab), lal_cmi.labels_rle(lab))"
"""
from __future__ import annotations

import importlib
import logging
import math
import os
import sys
from typing import Any, Sequence, Union

import numpy as np
import torch

from . import paths
from .common import text_cmi, token_langs  # noqa: F401  (single implementation lives in common.py)

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATE_DICT_FORMAT = "cmi_dpo_lal_state_dict_v1"   # 'format' key written by lal/export_state_dict.py
ZH, EN, BLANK, OTHER = 0, 1, 2, 3
CLASS_NAMES: tuple[str, ...] = ("zh", "en", "blank", "other")
N_CLASSES = len(CLASS_NAMES)
COUNT_KEYS: tuple[str, ...] = ("n_zh", "n_en", "n_blank", "n_other")   # TSV column names, class order
SAMPLE_RATE = 16000
N_MELS = 80
N_MEL_FRAMES = 3000          # 30 s window at 10 ms hop
N_ENC_FRAMES = 1500          # encoder frames per window (conv stride 2)
FRAME_SEC = 0.02             # seconds per encoder frame
MAX_AUDIO_SEC = float(N_ENC_FRAMES * FRAME_SEC)   # 30.0

_DEBUG_ENV_KEYS: tuple[str, ...] = ("CUDA_LAUNCH_BLOCKING", "TORCH_SHOW_CPP_STACKTRACES")


def lal_code_dir() -> str:
    """Dir holding WhisperLAL.py (CMI_DPO_LAL_CODE_DIR, default the vendored ``<repo>/lal``)."""
    return paths.lal_code_dir()


def lal_ckpt_default() -> str:
    """Configured LAL checkpoint (CMI_DPO_LAL_CKPT; SystemExit with a hint when unset)."""
    return paths.lal_ckpt()


def __getattr__(name: str) -> Any:
    """Lazy module attributes (PEP 562): ``LAL_CKPT_DEFAULT`` (CMI_DPO_LAL_CKPT, '' when unset,
    so that ``--help`` of the scripts works without the CMI_DPO_* variables) and ``LAL_ROOT``
    (= ``lal_code_dir()``). Neither is evaluated at import time."""
    if name == "LAL_CKPT_DEFAULT":
        return paths.get("LAL_CKPT", "") or ""
    if name == "LAL_ROOT":
        return lal_code_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def _import_whisperlal() -> Any:
    """Import the reference module ``WhisperLAL`` (class for unpickling / rebuilding) and return it.

    WhisperLAL.py (verbatim reference code) has import-time side effects: it sets
    CUDA_LAUNCH_BLOCKING=1 and TORCH_SHOW_CPP_STACKTRACES=1 in os.environ and enables
    ``torch.autograd.set_detect_anomaly(True)``. Both are undone here right after the
    import (environment variables are restored to their previous values).
    """
    code_dir = lal_code_dir()
    saved = {k: os.environ.get(k) for k in _DEBUG_ENV_KEYS}
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    module = importlib.import_module("WhisperLAL")
    torch.autograd.set_detect_anomaly(False)
    for key, old in saved.items():
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old
    LOG.info("imported WhisperLAL from %s (anomaly detection off, debug env restored)", code_dir)
    return module


def _torch_load_cpu(path: str) -> Any:
    """``torch.load(path, map_location='cpu')`` that also unpickles whole modules on torch >= 2.6
    (whose default became weights_only=True); the kwarg is dropped on a torch without it."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _rebuild_from_state_dict(obj: dict[str, Any], ckpt: str, whisperlal: Any,
                             base_model: Union[str, None]) -> torch.nn.Module:
    """``WhisperWithLAL(base_model, layer_index)`` + ``load_state_dict(strict=True)`` from an exported dict."""
    for key in ("state_dict", "layer_index", "n_lang", "d_model"):
        if key not in obj:
            raise ValueError(f"{ckpt}: state-dict file lacks key {key!r}")
    # precedence: explicit argument > CMI_DPO_LAL_BASE_MODEL (when set and non-empty) > the id stored by
    # lal/export_state_dict.py > the package default; the variable is the only way an offline user can
    # redirect an exported file to a local HF dir without re-exporting it
    configured = paths.get("LAL_BASE_MODEL")
    if base_model:
        base, source = base_model, "base_model argument"
    elif configured:
        base, source = configured, "CMI_DPO_LAL_BASE_MODEL"
    elif obj.get("base_model"):
        base, source = str(obj["base_model"]), "state-dict entry base_model"
    else:
        base, source = paths.lal_base_model(), "package default"
    layer_index = int(obj["layer_index"])
    LOG.info("rebuilding WhisperWithLAL(%s, layer_index=%d) for state dict %s (base model from %s)",
             base, layer_index, ckpt, source)
    model = whisperlal.WhisperWithLAL(base, layer_index=layer_index)
    if int(model.whisper.config.d_model) != int(obj["d_model"]):
        raise ValueError(f"{ckpt}: d_model {obj['d_model']} does not match base model {base} "
                         f"(d_model {model.whisper.config.d_model})")
    if int(model.language_cls.out_features) != int(obj["n_lang"]):
        raise ValueError(f"{ckpt}: n_lang {obj['n_lang']} does not match language_cls "
                         f"({model.language_cls.out_features} classes)")
    model.load_state_dict(obj["state_dict"], strict=True)
    return model


def load_lal_model(ckpt: str = "", device: Union[str, torch.device] = "cuda",
                   base_model: Union[str, None] = None) -> torch.nn.Module:
    """Load a ``WhisperWithLAL`` checkpoint in eval mode with gradients disabled.

    ``ckpt`` is either (a) a whole-pickled model (``torch.save(model)``, needs the training-time
    torch/transformers) or (b) a state-dict file from lal/export_state_dict.py: a dict with keys
    state_dict / base_model / layer_index / n_lang / d_model and ``format ==
    'cmi_dpo_lal_state_dict_v1'``, from which ``WhisperLAL.WhisperWithLAL(base_model,
    layer_index)`` is rebuilt (``base_model`` argument > CMI_DPO_LAL_BASE_MODEL when set >
    the file's ``base_model`` entry > ``openai/whisper-small``; an HF id works offline only when
    cached, otherwise point the variable at a local HF dir) and the weights are
    loaded with strict=True. The format is detected after ``torch.load(map_location='cpu')``.
    An empty ``ckpt`` means the configured CMI_DPO_LAL_CKPT.

    The CUDA context is created *before* ``WhisperLAL`` is imported so that the
    CUDA_LAUNCH_BLOCKING=1 it sets at import time never reaches the CUDA runtime.
    """
    ckpt = ckpt or lal_ckpt_default()
    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"device {dev} requested but CUDA is not available")
        torch.zeros(1, device=dev)          # initialise the CUDA context on this device now
    whisperlal = _import_whisperlal()
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)
    obj = _torch_load_cpu(ckpt)
    if isinstance(obj, dict) and obj.get("format"):
        if obj.get("format") != STATE_DICT_FORMAT:
            raise ValueError(f"{ckpt}: unknown state-dict format {obj.get('format')!r} (expected {STATE_DICT_FORMAT})")
        model = _rebuild_from_state_dict(obj, ckpt, whisperlal, base_model)
        kind = "state_dict"
    else:
        model = obj
        kind = "pickle"
    if not hasattr(model, "whisper") or not hasattr(model, "language_cls"):
        raise TypeError(f"{ckpt} did not load as a WhisperWithLAL model (got {type(model).__name__})")
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    LOG.info("loaded %s (%s, %s, %.1fM params, d_model=%d) on %s", ckpt, kind, type(model).__name__,
             n_params / 1e6, model.whisper.config.d_model, dev)
    return model


def model_device(model: torch.nn.Module) -> torch.device:
    """Device of the model's parameters."""
    return next(model.parameters()).device


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def whisper_mel(audio16k: np.ndarray, device: Union[str, torch.device, None] = None) -> torch.Tensor:
    """80-bin Whisper log-mel spectrogram of 16 kHz audio padded/trimmed to 30 s -> [80, 3000].

    Uses openai-whisper (``pad_or_trim`` + ``log_mel_spectrogram(n_mels=80)``), which is
    what the HF ``WhisperFeatureExtractor`` used at training time computes. ``device``
    optionally runs the STFT there (the result stays on that device).
    Raises ValueError when the audio holds NaN/Inf samples (they would poison the whole mel).
    """
    import whisper  # openai-whisper; imported lazily so the pure helpers work without it

    audio = np.ascontiguousarray(np.asarray(audio16k, dtype=np.float32).reshape(-1))
    if not np.isfinite(audio).all():
        # one NaN/Inf sample makes log_mel_spectrogram's `log_spec.max() - 8.0` clamp the whole
        # [80,3000] mel to NaN and the encoder/argmax then produce meaningless labels
        n_bad = int((~np.isfinite(audio)).sum())
        raise ValueError(f"audio contains {n_bad} non-finite (NaN/Inf) samples")
    audio = whisper.pad_or_trim(audio)                 # -> exactly 480000 samples
    mel = whisper.log_mel_spectrogram(audio, n_mels=N_MELS, device=device)
    if tuple(mel.shape) != (N_MELS, N_MEL_FRAMES):
        raise RuntimeError(f"unexpected mel shape {tuple(mel.shape)}, expected {(N_MELS, N_MEL_FRAMES)}")
    return mel


def n_valid_frames(duration_s: float) -> int:
    """Encoder frames that cover ``duration_s`` seconds: ceil(dur / 0.02), clipped to [0, 1500].

    The ratio is rounded to 6 decimals before ceil so that e.g. 1.1 s -> 55 frames and not
    56 (1.1 / 0.02 = 55.00000000000001 in binary floating point).
    """
    if duration_s <= 0.0:
        return 0
    n = math.ceil(round(duration_s / FRAME_SEC, 6))
    return max(0, min(N_ENC_FRAMES, n))


# ---------------------------------------------------------------------------
# Pseudo labels
# ---------------------------------------------------------------------------
@torch.no_grad()
def frame_language_logits(model: torch.nn.Module, mel_batch: torch.Tensor) -> torch.Tensor:
    """Encoder last_hidden_state -> language_cls -> [B, 1500, 4] logits (on the model device)."""
    if mel_batch.dim() == 2:
        mel_batch = mel_batch.unsqueeze(0)
    if mel_batch.dim() != 3 or tuple(mel_batch.shape[1:]) != (N_MELS, N_MEL_FRAMES):
        raise ValueError(f"mel_batch must be [B, {N_MELS}, {N_MEL_FRAMES}], got {tuple(mel_batch.shape)}")
    ref = next(model.parameters())
    mel = mel_batch.to(device=ref.device, dtype=ref.dtype)
    enc = model.whisper.model.encoder(mel).last_hidden_state      # [B, 1500, d_model]
    logits = model.language_cls(enc)                             # [B, 1500, 4]
    if tuple(logits.shape[1:]) != (N_ENC_FRAMES, N_CLASSES):
        raise RuntimeError(f"unexpected logits shape {tuple(logits.shape)}")
    return logits


def frame_language_labels(model: torch.nn.Module, mel_batch: torch.Tensor,
                          durations_s: Sequence[float]) -> list[np.ndarray]:
    """Per-utterance argmax frame labels, truncated to the frames covered by each duration.

    Returns a list of int64 arrays of length ``n_valid_frames(dur)`` (<= 1500) with values
    in {0: zh, 1: en, 2: blank, 3: other}.
    """
    if mel_batch.dim() == 2:
        mel_batch = mel_batch.unsqueeze(0)
    if len(durations_s) != mel_batch.shape[0]:
        raise ValueError(f"{len(durations_s)} durations for a batch of {mel_batch.shape[0]} mels")
    labels = frame_language_logits(model, mel_batch).argmax(dim=-1).to(torch.int64).cpu().numpy()
    return [labels[i, : n_valid_frames(float(d))].copy() for i, d in enumerate(durations_s)]


# ---------------------------------------------------------------------------
# CMI
# ---------------------------------------------------------------------------
def count_labels(labels: Union[np.ndarray, Sequence[int]]) -> dict[str, int]:
    """Frame counts ``{'n_frames', 'n_zh', 'n_en', 'n_blank', 'n_other'}`` of a label array."""
    arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if arr.size and (arr.min() < 0 or arr.max() >= N_CLASSES):
        raise ValueError(f"labels must be in [0, {N_CLASSES - 1}], got range [{arr.min()}, {arr.max()}]")
    bins = np.bincount(arr, minlength=N_CLASSES)
    counts: dict[str, int] = {"n_frames": int(arr.size)}
    for k, key in enumerate(COUNT_KEYS):
        counts[key] = int(bins[k])
    return counts


def cmi_from_labels(labels: Union[np.ndarray, Sequence[int]],
                    langs: Sequence[int] = (ZH, EN)) -> tuple[float, dict[str, int]]:
    """CMIspeech = (T - max_k T_k) / T over the classes in ``langs`` (0.0 when T == 0).

    Returns ``(cmi in [0, 1], counts)`` where counts holds n_frames, the four class counts
    and ``n_counted`` (= T, the frames belonging to ``langs``).
    """
    if not langs:
        raise ValueError("langs must name at least one class")
    counts = count_labels(labels)
    per_class = [counts[key] for key in COUNT_KEYS]
    t_k = [per_class[int(k)] for k in langs]
    total = sum(t_k)
    counts["n_counted"] = total
    cmi = 0.0 if total == 0 else (total - max(t_k)) / total
    return float(cmi), counts


def delta_cmi(cmi_synth: float, cmi_gt: float) -> float:
    """ΔCMI = |CMIspeech(synth) - CMIspeech(gt)|."""
    return abs(float(cmi_synth) - float(cmi_gt))


# ---------------------------------------------------------------------------
# Label run-length encoding (compact TSV dump)
# ---------------------------------------------------------------------------
def labels_rle(labels: Union[np.ndarray, Sequence[int]]) -> str:
    """Run-length encode labels as ``'0x120,1x35,2x10'`` (empty string for no frames)."""
    arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return ""
    change = np.flatnonzero(np.diff(arr)) + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [arr.size])))
    return ",".join(f"{int(arr[s])}x{int(n)}" for s, n in zip(starts, lengths))


def rle_to_labels(rle: str) -> np.ndarray:
    """Inverse of ``labels_rle``: ``'0x120,1x35'`` -> int64 array of 155 labels."""
    rle = rle.strip()
    if not rle:
        return np.zeros(0, dtype=np.int64)
    parts: list[np.ndarray] = []
    for run in rle.split(","):
        value, count = run.split("x")
        parts.append(np.full(int(count), int(value), dtype=np.int64))
    return np.concatenate(parts)


def summarize_counts(counts: dict[str, Any]) -> str:
    """One-line human-readable rendering of a counts dict (for logs)."""
    return " ".join(f"{name}={int(counts[key])}" for name, key in zip(CLASS_NAMES, COUNT_KEYS))
