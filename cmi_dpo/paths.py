"""Machine-specific locations for the cmi_dpo package, read from the environment.

Purpose
    Every path or site setting that differs between machines (CosyVoice checkout, model dirs,
    checkpoints, data root, optional launcher settings) is an environment variable prefixed
    ``CMI_DPO_``. A missing required variable raises SystemExit with a message telling the
    user to export ``CMI_DPO_<name>``.

    In addition, an optional KEY=VALUE file config/paths.env is read if present (by
    ``load_env_file``, ONCE at import time, see the bottom of this module; skipped when
    ``CMI_DPO_NO_ENV_FILE=1``). Variables already present in ``os.environ`` always win over
    the file. Importing this module never fails because of a missing file or variable — only
    the ``require``-based accessors do, when they are called.

Keys (all prefixed CMI_DPO_ in the environment):
    COSY_ROOT COSY_MODEL_DIR SFT_CKPT LAL_CODE_DIR LAL_CKPT LAL_BASE_MODEL ASR_MODEL_DIR
    UTMOS_SOURCE UTMOS_REPO UTMOS_CKPT DATA_ROOT
    optional, for cluster launchers: CONDA_SH ENV_MAIN ENV_ASR ENV_LAL
    SLURM_PARTITION SLURM_EXCLUDE SLURM_ACCOUNT SLURM_EXTRA

Environment
    Pure python (stdlib only); CPU is enough and it runs in every project env.

Example
    python -c "import cmi_dpo.paths as p; print(p.describe())"
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

LOG = logging.getLogger(__name__)

#: Absolute path of the repository root (the parent directory of this ``cmi_dpo`` package dir).
REPO_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PREFIX = "CMI_DPO_"
DEFAULT_ENV_FILE: str = os.path.join(REPO_ROOT, "config", "paths.env")
EXAMPLE_ENV_FILE: str = os.path.join(REPO_ROOT, "config", "paths.env.example")

#: Every known key (without prefix) -> (required?, default shown by describe(), one-line meaning).
KNOWN_KEYS: list[tuple[str, bool, str, str]] = [
    ("COSY_ROOT", True, "", "CosyVoice code dir (contains cosyvoice/ and Matcha-TTS/)"),
    ("COSY_MODEL_DIR", True, "", "CosyVoice2-0.5B pretrained model dir"),
    ("SFT_CKPT", False, "", "stage-1 fine-tuned LLM checkpoint; empty = stock llm.pt"),
    ("LAL_CODE_DIR", False, os.path.join(REPO_ROOT, "lal"), "dir holding WhisperLAL.py (unpickling / rebuild)"),
    ("LAL_CKPT", True, "", "Whisper-LAL checkpoint (pickled model or exported state dict)"),
    ("LAL_BASE_MODEL", False, "openai/whisper-small", "HF id or local dir of the LAL base Whisper model (set = overrides the id stored in an exported state dict)"),
    ("ASR_MODEL_DIR", True, "", "HF dir of the MER critic ASR model"),
    ("UTMOS_SOURCE", False, "hub", "'hub' (torch.hub SpeechMOS) or 'local' (UTMOS_REPO + UTMOS_CKPT)"),
    ("UTMOS_REPO", False, "", "local SpeechMOS checkout (CMI_DPO_UTMOS_SOURCE=local)"),
    ("UTMOS_CKPT", False, "", "local utmos22_strong state dict (CMI_DPO_UTMOS_SOURCE=local)"),
    ("DATA_ROOT", False, "", "dir holding the SEAME Kaldi dirs (train/valid/devman/devsge)"),
    ("CONDA_SH", False, "", "optional, for cluster launchers: conda.sh to source"),
    ("ENV_MAIN", False, "", "optional, for cluster launchers: conda env for CosyVoice / DPO / UTMOS stages"),
    ("ENV_ASR", False, "", "optional, for cluster launchers: conda env for the MER critic"),
    ("ENV_LAL", False, "", "optional, for cluster launchers: conda env for the Whisper-LAL / CMI critic"),
    ("SLURM_PARTITION", False, "", "optional, for cluster launchers: partition"),
    ("SLURM_EXCLUDE", False, "", "optional, for cluster launchers: nodes to exclude"),
    ("SLURM_ACCOUNT", False, "", "optional, for cluster launchers: account"),
    ("SLURM_EXTRA", False, "", "optional, for cluster launchers: extra scheduler options"),
]


# ---------------------------------------------------------------------------
# env file
# ---------------------------------------------------------------------------
def _parse_line(line: str) -> Optional[tuple[str, str]]:
    """Parse one ``KEY=VALUE`` line; returns None for blank / comment / malformed lines.

    Tolerates a leading ``export ``, single or double quotes around the value and a trailing
    ``# comment`` (separated by whitespace) after a quoted or an unquoted value, as ``bash``
    would read the same line.
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if s.startswith("export "):
        s = s[len("export "):].strip()
    if "=" not in s:
        return None
    key, value = s.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key or any(c.isspace() for c in key):
        return None
    quoted = re.match(r"""^(['"])(.*?)\1(\s*(#.*)?)?$""", value)
    if quoted:                     # quoted value; anything after the closing quote is a comment
        value = quoted.group(2)
    else:
        for marker in (" #", "\t#"):
            idx = value.find(marker)
            if idx >= 0:
                value = value[:idx].rstrip()
    return key, value


def load_env_file(path: Optional[str] = None) -> dict[str, str]:
    """Read ``KEY=VALUE`` lines from ``path`` (default ``DEFAULT_ENV_FILE``, see the module docstring).

    Values already present in ``os.environ`` WIN over the file; keys not yet set are exported
    to ``os.environ``. Returns the merged mapping for the keys found in the file (environment
    value when it was already set, file value otherwise). A missing file silently gives ``{}``.
    """
    path = path or DEFAULT_ENV_FILE
    merged: dict[str, str] = {}
    if not os.path.isfile(path):
        return merged
    with open(path, encoding="utf-8") as f:
        for raw in f:
            parsed = _parse_line(raw)
            if parsed is None:
                continue
            key, value = parsed
            if key in os.environ:
                merged[key] = os.environ[key]
            else:
                os.environ[key] = value
                merged[key] = value
    LOG.debug("loaded %d keys from %s", len(merged), path)
    return merged


# ---------------------------------------------------------------------------
# accessors
# ---------------------------------------------------------------------------
def _expand(value: str) -> str:
    """Expand ``~`` and ``$VAR`` / ``${VAR}`` references in a value."""
    return os.path.expandvars(os.path.expanduser(value))


def get(name: str, default: Optional[str] = None) -> Optional[str]:
    """Value of ``CMI_DPO_<name>`` (``~`` and ``$VARS`` expanded); ``default`` when unset or empty."""
    raw = os.environ.get(ENV_PREFIX + name)
    if raw is None or raw.strip() == "":
        return default
    return _expand(raw.strip())


def require(name: str) -> str:
    """Like ``get`` but raises SystemExit with a helpful message when the variable is unset or empty."""
    value = get(name)
    if not value:
        var = ENV_PREFIX + name
        raise SystemExit(
            f"{var} is not set (or empty). Export it in the environment with the path for this "
            f"machine, e.g. export {var}=/path/on/this/machine"
        )
    return value


def cosy_root() -> str:
    """CosyVoice code dir that contains cosyvoice/ and Matcha-TTS/ (required)."""
    return require("COSY_ROOT")


def cosy_model_dir() -> str:
    """CosyVoice2-0.5B pretrained model dir (required)."""
    return require("COSY_MODEL_DIR")


def sft_ckpt() -> str:
    """Optional stage-1 LLM checkpoint; '' means the stock llm.pt."""
    return get("SFT_CKPT", "") or ""


def lal_code_dir() -> str:
    """Dir holding WhisperLAL.py (default: the vendored ``<repo>/lal``)."""
    return get("LAL_CODE_DIR", os.path.join(REPO_ROOT, "lal")) or os.path.join(REPO_ROOT, "lal")


def lal_ckpt() -> str:
    """Whisper-LAL checkpoint: pickled WhisperWithLAL or exported state-dict file (required)."""
    return require("LAL_CKPT")


def lal_base_model() -> str:
    """HF id (cached offline) or local HF dir of the LAL base Whisper model (default openai/whisper-small).

    Package default only: ``lal_cmi.load_lal_model`` prefers an explicit argument, then the variable when
    it is set, then the ``base_model`` entry of an exported state dict, before falling back to this.
    """
    return get("LAL_BASE_MODEL", "openai/whisper-small") or "openai/whisper-small"


def asr_model_dir() -> str:
    """HF dir of the MER critic ASR model (required)."""
    return require("ASR_MODEL_DIR")


def utmos_source() -> str:
    """'hub' (torch.hub SpeechMOS) or 'local' (UTMOS_REPO + UTMOS_CKPT); default 'hub'."""
    return get("UTMOS_SOURCE", "hub") or "hub"


def utmos_repo() -> str:
    """Local SpeechMOS checkout ('' when unset)."""
    return get("UTMOS_REPO", "") or ""


def utmos_ckpt() -> str:
    """Local utmos22_strong state-dict file ('' when unset)."""
    return get("UTMOS_CKPT", "") or ""


def data_root() -> str:
    """Dir with the SEAME Kaldi dirs ('' when unset)."""
    return get("DATA_ROOT", "") or ""


def describe() -> str:
    """Multi-line listing of every known key with its current value (for --show_paths flags)."""
    lines = [f"REPO_ROOT = {REPO_ROOT}",
             f"env file  = {DEFAULT_ENV_FILE} ({'present' if os.path.isfile(DEFAULT_ENV_FILE) else 'missing'})"]
    for name, required, default, meaning in KNOWN_KEYS:
        value = get(name)
        if value is None:
            shown = f"<unset>{' (REQUIRED)' if required else ''}"
            if default:
                shown += f" -> default {default}"
        else:
            shown = value
        lines.append(f"{ENV_PREFIX}{name:<16} = {shown}    # {meaning}")
    return "\n".join(lines)


if os.environ.get("CMI_DPO_NO_ENV_FILE") != "1":
    load_env_file()
