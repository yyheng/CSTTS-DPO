"""Shared, dependency-light helpers for the cmi_dpo package.

Purpose
    Kaldi-dir readers, manifest / TSV IO, SEAME text conventions (text_tts, text_ref),
    the SEAME MER normaliser (faithful copy of ``make_seame_normalizer`` from the SEAME
    Whisper baselines' ``score.py``, opencc optional), a jiwer-free Levenshtein
    distance, audio loading at 16 kHz, seeding, sharding and token-level text CMI.
    Critic plumbing shared by 11_/12_/13_score_*: ``repair_torn_tsv`` (truncate a torn
    last line before --resume appends), ``append_failed`` / ``reset_failed``
    (``<out>.failed.txt``) and ``failure_exit_code`` (--max_fail_frac / --strict policy).

Environment
    Pure python + numpy; runs in every project env (cosyvoicenew, asr-whisper,
    whisperold). torch/torchaudio/soundfile/opencc are imported lazily and optionally.

Example (inside an srun/sbatch shell on a compute node)
    python -c "from cmi_dpo import common; print(common.seame_normalize('我 跟 你 讲 <noise> don\\'t 吧'))"
"""
from __future__ import annotations

import logging
import os
import random
import re
from typing import Any, Callable, Iterable, Optional

import numpy as np

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MANIFEST_COLUMNS: list[str] = [
    "utt", "spk", "dur", "wav", "text_raw", "text_tts", "text_ref",
    "prompt_utt", "prompt_wav", "prompt_text",
]

# CJK ranges: BMP + ext-A + compatibility (same as baselines/src/score.py).
_CJK = re.compile(r"([㐀-䶿一-鿿豈-﫿])")
_TAG = re.compile(r"<[^>]*>")          # <noise>, <unk>, <UNK> ...
_APOS = re.compile(r"['’]")            # don't -> dont
_ASCII_LETTER = re.compile(r"[A-Za-z]")
_OPENCC_STATE: dict[str, Any] = {"loaded": False, "convert": None}


# ---------------------------------------------------------------------------
# Kaldi readers
# ---------------------------------------------------------------------------
def _read_two_col(path: str) -> dict[str, str]:
    """Read a Kaldi ``<key> <rest of line>`` file into an ordered dict."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            out[parts[0]] = parts[1].strip() if len(parts) > 1 else ""
    return out


def read_kaldi_dir(d: str) -> dict[str, dict[str, Any]]:
    """Read wav.scp / text / utt2spk / utt2dur from a Kaldi directory.

    Returns ``{utt: {'wav': str, 'text_raw': str, 'spk': str, 'dur': float}}`` keyed
    in wav.scp order, restricted to utterances present in both wav.scp and text.
    ``text_raw`` is the transcript with tokens joined by single spaces. Missing
    utt2spk falls back to the utt-id prefix before the first '-'; missing utt2dur
    falls back to reading the audio header with soundfile.
    """
    wav = _read_two_col(os.path.join(d, "wav.scp"))
    text = _read_two_col(os.path.join(d, "text"))
    spk_path = os.path.join(d, "utt2spk")
    dur_path = os.path.join(d, "utt2dur")
    spk = _read_two_col(spk_path) if os.path.exists(spk_path) else {}
    dur = _read_two_col(dur_path) if os.path.exists(dur_path) else {}
    if not spk:
        LOG.warning("%s missing: speaker = utt prefix before '-'", spk_path)
    if not dur:
        LOG.warning("%s missing: durations read from audio headers (slow)", dur_path)

    out: dict[str, dict[str, Any]] = {}
    n_no_text = 0
    for utt, path in wav.items():
        if utt not in text:
            n_no_text += 1
            continue
        if utt in dur:
            d_s = float(dur[utt])
        else:
            import soundfile as sf  # optional dependency, only on this fallback path
            info = sf.info(path)
            d_s = float(info.frames) / float(info.samplerate)
        out[utt] = {
            "wav": path,
            "text_raw": " ".join(text[utt].split()),
            "spk": spk.get(utt, utt.split("-", 1)[0]),
            "dur": d_s,
        }
    if n_no_text:
        LOG.warning("%d utts in wav.scp have no transcript in text; dropped", n_no_text)
    LOG.info("read_kaldi_dir(%s): %d utts, %.2f h", d, len(out), sum(v["dur"] for v in out.values()) / 3600.0)
    return out


# ---------------------------------------------------------------------------
# Text conventions
# ---------------------------------------------------------------------------
def strip_tags(text: str) -> str:
    """text_tts: drop ``<...>`` tags and collapse whitespace; SEAME spacing is kept."""
    return " ".join(_TAG.sub(" ", text).split())


def _opencc_convert() -> Optional[Callable[[str], str]]:
    """Return the opencc t2s converter if opencc is importable, else None (cached)."""
    if not _OPENCC_STATE["loaded"]:
        _OPENCC_STATE["loaded"] = True
        try:
            import opencc  # type: ignore
            _OPENCC_STATE["convert"] = opencc.OpenCC("t2s").convert
            LOG.info("seame_normalize: opencc t2s enabled")
        except Exception:  # noqa: BLE001 - any failure means "no opencc"
            _OPENCC_STATE["convert"] = None
            LOG.warning("seame_normalize: opencc NOT available in this env -> no t2s conversion; "
                        "traditional Han characters stay as-is (env-dependent output)")
    return _OPENCC_STATE["convert"]


def opencc_available() -> bool:
    """True when opencc t2s is importable here (i.e. seame_normalize output matches the asr-whisper env)."""
    return _opencc_convert() is not None


def seame_normalize(text: str) -> str:
    """SEAME MER normaliser (copy of baselines/src/score.py make_seame_normalizer).

    lower-case; ``<tag>`` -> space; opencc t2s if importable; apostrophes removed;
    every Han character isolated by spaces; remaining punctuation -> space;
    '_' -> space; whitespace collapsed. Result: CJK per character, Latin per word.

    NOTE: the output is env-dependent (t2s only where opencc is importable, e.g.
    asr-whisper; NOT cosyvoicenew/whisperold). Reference and hypothesis must be
    normalised in the same env; see ``opencc_available()``.
    """
    convert = _opencc_convert()
    s = text.lower()
    s = _TAG.sub(" ", s)
    if convert:
        s = convert(s)
    s = _APOS.sub("", s)
    s = _CJK.sub(r" \1 ", s)           # isolate each Han char
    s = re.sub(r"[^\w ]", " ", s, flags=re.UNICODE)  # drop punctuation
    s = s.replace("_", " ")
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# Edit distance / MER
# ---------------------------------------------------------------------------
def levenshtein(a: list[str], b: list[str]) -> int:
    """Token-level Levenshtein distance, standard O(len(a)*len(b)) two-row DP."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, tb in enumerate(b, start=1):
            cost = 0 if ta == tb else 1
            cur[j] = min(prev[j] + 1,          # deletion
                         cur[j - 1] + 1,       # insertion
                         prev[j - 1] + cost)   # substitution / match
        prev = cur
    return prev[-1]


def mer_pair(ref: str, hyp: str) -> tuple[int, int, float]:
    """(edits, n_ref, mer) for two ALREADY-NORMALISED strings (whitespace tokenised).

    mer = edits / n_ref and may exceed 1.0. When n_ref == 0 the ratio is undefined;
    we return mer = float(edits) (i.e. every hyp token counts as an insertion).
    """
    r = ref.split()
    h = hyp.split()
    edits = levenshtein(r, h)
    n_ref = len(r)
    mer = edits / n_ref if n_ref > 0 else float(edits)
    return edits, n_ref, mer


# ---------------------------------------------------------------------------
# TSV / manifest IO
# ---------------------------------------------------------------------------
def _clean_cell(v: Any) -> str:
    """Render a value as a single TSV cell (no tabs / newlines)."""
    if isinstance(v, float):
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        if s in ("", "-", "-0"):
            s = "0"
    else:
        s = "" if v is None else str(v)
    return s.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def read_tsv(path: str, strict: bool = False) -> list[dict[str, str]]:
    """Read a header-ed TSV into a list of dicts (all values are strings).

    A malformed (wrong column count) LAST line is the signature of a job killed
    mid-``append_tsv``; with ``strict=False`` (default) it is dropped with a warning
    so resumable scripts can recover (the row is simply recomputed). A malformed
    line anywhere else, or any malformed line with ``strict=True``, raises ValueError.
    """
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            return rows
        cols = header_line.rstrip("\n").split("\t")
        lines = [(ln, line.rstrip("\n")) for ln, line in enumerate(f, start=2)]
    while lines and not lines[-1][1]:
        lines.pop()  # trailing empty lines
    for idx, (ln, line) in enumerate(lines):
        if not line:
            continue
        vals = line.split("\t")
        if len(vals) != len(cols):
            msg = f"{path}:{ln}: expected {len(cols)} columns, got {len(vals)}"
            if strict or idx != len(lines) - 1:
                raise ValueError(msg)
            LOG.warning("%s -> partial last line dropped (interrupted append)", msg)
            break
        rows.append(dict(zip(cols, vals)))
    return rows


def write_tsv(path: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write rows (dicts) as a header-ed TSV with exactly ``columns`` in that order."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for r in rows:
            f.write("\t".join(_clean_cell(r.get(c)) for c in columns) + "\n")
    os.replace(tmp, path)


def append_tsv(path: str, row: dict[str, Any], columns: list[str]) -> None:
    """Append one row; writes the header first if the file is missing or empty."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    need_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8") as f:
        if need_header:
            f.write("\t".join(columns) + "\n")
        f.write("\t".join(_clean_cell(row.get(c)) for c in columns) + "\n")
        f.flush()


def repair_torn_tsv(path: str, n_columns: int) -> int:
    """Truncate a last line that a hard kill left incomplete; return the number of bytes removed.

    ``append_tsv`` writes one row per open/write/flush, so a SIGKILL (time limit, OOM) can
    leave the tail of a row on disk: a last line without a trailing newline, or one with the
    wrong number of columns. ``read_tsv`` merely drops such a line while reading, which is
    not enough for resumable critics: the next ``append_tsv`` would be glued onto it and
    corrupt BOTH rows. This helper truncates the file itself (``os.truncate``) to the start
    of the torn line so the file is safe to append to; the row is simply scored again.
    A torn header line (no newline yet) is cut back to an empty file, which ``append_tsv``
    then re-heads. A missing/empty file, a header-only file or a well-formed file is left
    untouched (returns 0). Shared by 11_score_mer / 12_score_utmos / 13_score_cmi.
    """
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        raw = f.read()
    if not raw:
        return 0
    complete = raw.endswith(b"\n")
    body = raw[:-1] if complete else raw
    start = body.rfind(b"\n") + 1            # byte offset of the last line (0 = header line)
    last = body[start:]
    if start == 0:
        if complete:
            return 0                         # header only: nothing to repair
        LOG.warning("%s: header line has no newline (%d bytes); file reset to empty", path, len(raw))
        os.truncate(path, 0)
        return len(raw)
    if not last:
        return 0
    n_cols = last.count(b"\t") + 1
    if complete and n_cols == n_columns:
        return 0
    removed = len(raw) - start
    LOG.warning("%s: last line is incomplete (%d bytes, %d/%d columns, newline=%s); truncating %d bytes, "
                "its row will be scored again", path, len(last), n_cols, n_columns, complete, removed)
    os.truncate(path, start)
    return removed


def append_failed(path: str, utt: str, cand: str, wav: str, reason: str) -> None:
    """Append one ``utt cand wav reason`` line to a critic's ``<out>.failed.txt`` (created on demand)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{_clean_cell(utt)}\t{_clean_cell(cand)}\t{_clean_cell(wav)}\t{_clean_cell(reason)}\n")


def reset_failed(path: str) -> None:
    """Remove a stale ``<out>.failed.txt`` before a (re)run: the rows it lists are re-attempted anyway."""
    if os.path.exists(path):
        os.remove(path)


def failure_exit_code(n_failed: int, n_total: int, max_fail_frac: float, strict: bool,
                      failed_txt: str, log: Optional[logging.Logger] = None) -> int:
    """Shared critic exit-code policy (11_/12_/13_score_*): 0 when the failures are tolerable, else 2.

    ``n_failed`` rows out of ``n_total`` in scope (rows already done under --resume count in
    the denominator) could not be scored and are listed in ``failed_txt``. The failed fraction
    ``n_failed / max(n_total, 1)`` is compared with ``max_fail_frac``: at or below it a
    WARNING is logged and 0 is returned so an ``afterok`` dependency chain (and the sbatch
    ``.done`` marker) proceeds; above it, or with ``strict`` and any failure, an ERROR is
    logged and 2 is returned. Re-running with --resume retries exactly the failed rows.
    """
    log = log or LOG
    if n_failed <= 0:
        return 0
    frac = n_failed / max(n_total, 1)
    if strict or frac > max_fail_frac:
        log.error("%d/%d rows (%.2f %%) could not be scored (listed in %s); %s -> exit 2; "
                  "re-run with --resume to retry them", n_failed, n_total, 100.0 * frac, failed_txt,
                  "--strict" if strict else f"above --max_fail_frac {max_fail_frac:g}")
        return 2
    log.warning("%d/%d rows (%.2f %%) could not be scored (listed in %s); within --max_fail_frac %g -> "
                "exit 0 (pipeline continues without them); re-run with --resume to retry",
                n_failed, n_total, 100.0 * frac, failed_txt, max_fail_frac)
    return 0


def read_manifest(path: str) -> list[dict[str, Any]]:
    """Read a manifest TSV (see MANIFEST_COLUMNS); ``dur`` is cast to float."""
    rows = read_tsv(path)
    missing = [c for c in MANIFEST_COLUMNS if rows and c not in rows[0]]
    if missing:
        raise ValueError(f"{path}: manifest missing columns {missing}")
    out: list[dict[str, Any]] = []
    for r in rows:
        r2: dict[str, Any] = dict(r)
        r2["dur"] = float(r["dur"])
        out.append(r2)
    return out


def write_manifest(path: str, rows: list[dict[str, Any]]) -> None:
    """Write manifest rows with exactly MANIFEST_COLUMNS."""
    write_tsv(path, rows, MANIFEST_COLUMNS)


MANIFEST_HASH_COLUMNS: list[str] = ["utt", "text_raw", "text_tts", "prompt_utt", "prompt_wav", "prompt_text"]


def manifest_content_hash(rows: Iterable[dict[str, Any]]) -> str:
    """sha256 over the columns that condition generation (MANIFEST_HASH_COLUMNS), ORDER-INDEPENDENT.

    Used as the ``manifest_sha256`` entry of the gen/synth config fingerprints: a manifest
    re-generated at the same path with another prompt assignment / --seed / --text_mode changes
    the hash, whereas a re-ordering of the same rows (or a different text_ref / spk / dur / wav
    column) does not. Cells are TSV-cleaned exactly as write_tsv writes them.
    """
    import hashlib

    lines = sorted("\t".join(_clean_cell(r.get(c)) for c in MANIFEST_HASH_COLUMNS) for r in rows)
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
def resample(wav: "Any", orig: int, target: int) -> "Any":
    """Resample a torch tensor [1,T] (or [T]) with torchaudio.functional.resample."""
    if orig == target:
        return wav
    import torchaudio
    return torchaudio.functional.resample(wav, orig_freq=orig, new_freq=target)


def load_audio16k(path: str) -> np.ndarray:
    """Load any audio file as mono float32 at 16 kHz (soundfile if importable, else torchaudio)."""
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # [T, C]
        mono = data.mean(axis=1).astype(np.float32)
    except ImportError:
        import torchaudio
        wav, sr = torchaudio.load(path)  # [C, T]
        mono = wav.mean(dim=0).numpy().astype(np.float32)
    if sr != 16000:
        import torch
        t = torch.from_numpy(mono).unsqueeze(0)
        mono = resample(t, int(sr), 16000).squeeze(0).numpy().astype(np.float32)
    return np.ascontiguousarray(mono)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def seed_all(seed: int) -> None:
    """Seed python ``random``, numpy and (if importable) torch / CUDA.

    Str-hash order is NOT covered (PYTHONHASHSEED is read only at interpreter
    start-up; export it in the sbatch header if needed). Manifest / prompt draws
    are deterministic through ``random.Random(seed)`` plus sorted utt order.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def shard(rows: list, i: int, n: int) -> list:
    """Return shard ``i`` of ``n`` (round-robin ``rows[i::n]``); shards are disjoint and cover all rows."""
    if n < 1 or not 0 <= i < n:
        raise ValueError(f"invalid shard {i}/{n}")
    return rows[i::n]


def token_langs(text_tts: str) -> list[int]:
    """Per-token language ids for text CMI: 0 = zh (contains a Han char), 1 = en (contains an ASCII letter); others skipped.

    ``<...>`` tags are removed first so the result is the same for --text_mode strip and raw.
    """
    langs: list[int] = []
    for tok in _TAG.sub(" ", text_tts).split():
        if _CJK.search(tok):
            langs.append(0)
        elif _ASCII_LETTER.search(tok):
            langs.append(1)
    return langs


def text_cmi(text_tts: str) -> float:
    """Token-level code-mixing index: (T - max_k T_k) / T over zh/en tokens; 0.0 if T == 0."""
    langs = token_langs(text_tts)
    total = len(langs)
    if total == 0:
        return 0.0
    n_zh = sum(1 for x in langs if x == 0)
    return (total - max(n_zh, total - n_zh)) / total


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging with timestamps (idempotent)."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def hours(rows: Iterable[dict[str, Any]]) -> float:
    """Total duration in hours of rows carrying a ``dur`` field."""
    return sum(float(r["dur"]) for r in rows) / 3600.0
