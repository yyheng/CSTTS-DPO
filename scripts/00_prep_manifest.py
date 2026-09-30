"""Build a TTS manifest TSV from a SEAME Kaldi directory (manifest format in docs/DESIGN.md).

Purpose
    For every target utterance write one row
        utt  spk  dur  wav  text_raw  text_tts  text_ref  prompt_utt  prompt_wav  prompt_text
    where
      text_tts    = the text the TTS LLM is conditioned on. --text_mode strip (default) drops
                    <noise>/<UNK> tags; --text_mode raw keeps text_raw verbatim, which is the
                    convention the authors' stage-1 model (SEAME_Final/epoch_020.pth, built by
                    TTS_finetune/CosyVoice/data_processing.py) was trained on. The same mode is
                    applied to prompt_text.
      text_ref    = seame_normalize(text_raw) (MER reference). It is written ONLY when opencc is
                    importable in this env (t2s applied, identical to the asr-whisper scorer);
                    otherwise the column is left EMPTY so that 11_score_mer recomputes it in the
                    scoring env (env-independent references).
      prompt_*    = the zero-shot speaker prompt:
      --prompt_mode same_spk : a different utterance of the same speaker whose duration lies
                               in [--prompt_min, --prompt_max] s, drawn with a seeded RNG;
                               fallback = the other utterance of the speaker (pool restricted to
                               [--min_dur, 30] s, the CosyVoice frontend limit) whose duration is
                               closest to the window. A target whose speaker has NO other pool
                               utterance has no prompt: by default it is DROPPED (count in the
                               log, one "utt<TAB>reason" line in <out stem>.dropped.txt).
                               --allow_self_prompt instead falls back to the utterance itself.
                               That makes prompt == target (same audio AND text, i.e. the
                               target leaks into the prompt), hence opt-in.
      --prompt_mode self     : every utterance is its own prompt (explicit experiment mode; the
                               leak above applies to all rows; nothing is dropped).
      --prompt_pool all      : (default) prompts are drawn from the WHOLE Kaldi dir (every utt
                               of the speaker inside the pool window), NOT only from the
                               --hours / --limit subset of targets. The log states this.
      --prompt_pool subset   : the pool is restricted to the selected targets, so a subset
                               manifest is self-contained (more drops for tiny subsets).
    Targets are filtered to [--min_dur, --max_dur] s and non-empty text after tag stripping.
    --hours draws a seeded random subset of targets until the cumulative duration reaches the
    cap (e.g. the paper's 100 h real subset). The log reports the hours available after the
    filters and says explicitly when the cap selects everything (SEAME train filtered to
    [1, 30] s is ~94.3 h, so --hours 100 selects all of it). --limit keeps the first N rows
    (sorted utt order) for smoke tests. Prompt drops happen AFTER --hours/--limit, so the
    written hours can be slightly below the cap; the log and json carry both numbers.

Outputs
    <out>                    the manifest TSV
    <out stem>.json          {n_utts, hours, n_speakers, n_dropped_no_prompt, n_available,
                              hours_available, hours_cap_selects_all, n_selected,
                              hours_selected, prompt_counts, mean_text_cmi_pct, args}
    <out stem>.dropped.txt   "utt<TAB>reason" per dropped target (always written, may be empty)
    <out stem> = --out without its extension (m.tsv -> m.json, m.dropped.txt), the same
    convention as 01_build_sft_cache.py.

Environment
    cosyvoicenew (pure python + numpy; no GPU; NO opencc -> text_ref left empty, see above).
    Run on a compute node via srun/sbatch.

Paths
    --data_dir defaults to $CMI_DPO_DATA_ROOT/train when CMI_DPO_DATA_ROOT is set (config/paths.env
    or the environment, see cmi_dpo.paths) and is required otherwise; --out defaults to
    <repo>/data/train_manifest.tsv. --show_paths prints the configured paths and exits.

Example
    python -u scripts/00_prep_manifest.py \
        --data_dir $CMI_DPO_DATA_ROOT/train \
        --out data/train_manifest.tsv --seed 0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmi_dpo import common, paths  # noqa: E402

LOG = logging.getLogger("00_prep_manifest")

DEFAULT_OUT = os.path.join(paths.REPO_ROOT, "data", "train_manifest.tsv")
PROMPT_HARD_MAX_S = 30.0  # cosyvoice/cli/frontend.py:81 asserts prompt audio <= 30 s
REASON_NO_PROMPT = "no_other_prompt_utt"


class ShowPathsAction(argparse.Action):
    """--show_paths: print cmi_dpo.paths.describe() and exit, before the required flags are checked."""

    def __init__(self, option_strings: list[str], dest: str, **kwargs: Any) -> None:
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS,
                         help=kwargs.get("help", "print the configured paths (cmi_dpo.paths) and exit"))

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
                 values: Any, option_string: str | None = None) -> None:
        print(paths.describe())
        parser.exit()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build the cmi_dpo manifest TSV (targets + speaker prompts) from a Kaldi dir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data_dir", default=None,
                    help="Kaldi dir with wav.scp, text, utt2spk, utt2dur, e.g. $CMI_DPO_DATA_ROOT/train "
                         "(default: $CMI_DPO_DATA_ROOT/train when CMI_DPO_DATA_ROOT is set, else required)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output manifest TSV path; <out stem>.json and <out stem>.dropped.txt are written next to it")
    ap.add_argument("--show_paths", action=ShowPathsAction)
    ap.add_argument("--text_mode", choices=["strip", "raw"], default="strip",
                    help="text_tts/prompt_text convention: strip = <...> tags removed; "
                         "raw = text_raw verbatim (tags kept, = the authors' stage-1 SFT convention)")
    ap.add_argument("--prompt_mode", choices=["same_spk", "self"], default="same_spk",
                    help="same_spk: another utt of the same speaker in the prompt duration window; "
                         "self: own utt (explicit experiment mode, prompt == target)")
    ap.add_argument("--prompt_pool", choices=["all", "subset"], default="all",
                    help="same_spk prompt pool: all = every utt of the whole Kaldi dir (not only the "
                         "--hours/--limit subset); subset = only the selected targets")
    ap.add_argument("--allow_self_prompt", action="store_true",
                    help="same_spk: fall back to the utterance itself when its speaker has no other "
                         "pool utt (prompt == target, leaks the target into the prompt). Default off: "
                         "such targets are dropped and listed in <out stem>.dropped.txt")
    ap.add_argument("--prompt_min", type=float, default=3.0, help="min prompt duration in seconds")
    ap.add_argument("--prompt_max", type=float, default=10.0, help="max prompt duration in seconds")
    ap.add_argument("--min_dur", type=float, default=1.0, help="drop targets shorter than this (s)")
    ap.add_argument("--max_dur", type=float, default=30.0, help="drop targets longer than this (s)")
    ap.add_argument("--hours", type=float, default=None,
                    help="cap: seeded random subset of targets until cumulative duration >= hours "
                         "(a cap >= the available hours selects everything)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for subset selection and prompt draws")
    ap.add_argument("--limit", type=int, default=None, help="keep only the first N rows (smoke tests)")
    args = ap.parse_args(argv)
    if args.data_dir is None:  # resolved after parsing so that --help works without config/paths.env
        root = paths.data_root()
        if not root:
            ap.error("--data_dir is required (or set CMI_DPO_DATA_ROOT, then it defaults to $CMI_DPO_DATA_ROOT/train)")
        args.data_dir = os.path.join(root, "train")
    if args.prompt_min > args.prompt_max:
        ap.error(f"--prompt_min {args.prompt_min} > --prompt_max {args.prompt_max}")
    if args.min_dur > args.max_dur:
        ap.error(f"--min_dur {args.min_dur} > --max_dur {args.max_dur}")
    if args.prompt_max > PROMPT_HARD_MAX_S:
        ap.error(f"--prompt_max {args.prompt_max} > {PROMPT_HARD_MAX_S} s (CosyVoice frontend limit)")
    if args.hours is not None and args.hours <= 0:
        ap.error(f"--hours {args.hours} must be > 0")
    return args


def sidecar_path(out: str, suffix: str) -> str:
    """<out stem><suffix>: m.tsv -> m.json / m.dropped.txt (01_build_sft_cache convention)."""
    return os.path.splitext(out)[0] + suffix


def tts_text(text_raw: str, text_mode: str) -> str:
    """text_tts / prompt_text under --text_mode: 'strip' drops <...> tags, 'raw' keeps text_raw."""
    if text_mode == "raw":
        return " ".join(text_raw.split())
    return common.strip_tags(text_raw)


def select_targets(data: dict[str, dict[str, Any]], min_dur: float, max_dur: float) -> list[str]:
    """Utterance ids (data order) inside the duration window with non-empty text after tag stripping."""
    keep: list[str] = []
    n_dur = n_text = 0
    for utt, row in data.items():
        if not (min_dur <= row["dur"] <= max_dur):
            n_dur += 1
            continue
        if not common.strip_tags(row["text_raw"]):
            n_text += 1
            continue
        keep.append(utt)
    LOG.info("targets: %d kept (%.2f h available after filtering), %d outside [%.1f, %.1f] s, "
             "%d empty after tag stripping", len(keep), utt_hours(keep, data), n_dur, min_dur,
             max_dur, n_text)
    return keep


def utt_hours(utts: list[str], data: dict[str, dict[str, Any]]) -> float:
    """Cumulative duration of ``utts`` in hours."""
    return sum(float(data[u]["dur"]) for u in utts) / 3600.0


def subset_by_hours(utts: list[str], data: dict[str, dict[str, Any]], hours: float,
                    rng: random.Random) -> tuple[list[str], bool]:
    """Seeded random subset accumulated until the cumulative duration reaches ``hours``.

    Returns (picked, selects_all). ``selects_all`` is True when the cap is >= the available
    hours, i.e. the cap is not binding and every filtered target is kept (in shuffled order;
    the caller sorts).
    """
    available = utt_hours(utts, data)
    order = list(utts)
    rng.shuffle(order)
    total = 0.0
    picked: list[str] = []
    for utt in order:
        if total >= hours * 3600.0:
            break
        picked.append(utt)
        total += data[utt]["dur"]
    selects_all = len(picked) == len(utts)
    if selects_all:
        LOG.info("--hours %.2f >= %.2f h available after filtering: the cap selects EVERYTHING "
                 "(%d utts, %.2f h)", hours, available, len(picked), total / 3600.0)
    else:
        LOG.info("--hours %.2f of %.2f h available: picked %d / %d utts (%.2f h selected)",
                 hours, available, len(picked), len(utts), total / 3600.0)
    return picked, selects_all


def build_prompt_pools(data: dict[str, dict[str, Any]], pool_min: float,
                       pool_max: float = PROMPT_HARD_MAX_S) -> dict[str, list[str]]:
    """spk -> sorted list of utts usable as prompts: non-empty text after tag stripping and
    duration in [pool_min, pool_max] s (pool_max defaults to the 30 s CosyVoice frontend limit)."""
    pools: dict[str, list[str]] = {}
    n_dur = 0
    for utt, row in data.items():
        if not common.strip_tags(row["text_raw"]):
            continue
        if not (pool_min <= row["dur"] <= pool_max):
            n_dur += 1
            continue
        pools.setdefault(row["spk"], []).append(utt)
    for spk in pools:
        pools[spk].sort()
    LOG.info("prompt pools: %d speakers, %d utts; %d utts outside [%.2f, %.1f] s excluded",
             len(pools), sum(len(v) for v in pools.values()), n_dur, pool_min, pool_max)
    return pools


def _window_distance(dur: float, pmin: float, pmax: float) -> float:
    """0 inside [pmin, pmax], else distance to the nearest edge."""
    if dur < pmin:
        return pmin - dur
    if dur > pmax:
        return dur - pmax
    return 0.0


def choose_prompt(utt: str, spk: str, pools: dict[str, list[str]], data: dict[str, dict[str, Any]],
                  pmin: float, pmax: float, rng: random.Random) -> tuple[str | None, str]:
    """Return (prompt_utt, reason) with reason in {'window', 'any_other', 'none'}.

    'window'    : seeded uniform draw among the speaker's other utts with pmin <= dur <= pmax.
    'any_other' : no such utt; the speaker's other pool utt whose duration is closest to the
                  window (deterministic: ties broken by sorted utt id).
    'none'      : the speaker has no other pool utt (prompt_utt is None; the caller decides
                  between dropping the target and --allow_self_prompt).
    """
    pool = pools.get(spk, [])
    in_window = [u for u in pool if u != utt and pmin <= data[u]["dur"] <= pmax]
    if in_window:
        return rng.choice(in_window), "window"
    others = [u for u in pool if u != utt]
    if others:
        return min(others, key=lambda u: (_window_distance(data[u]["dur"], pmin, pmax), u)), "any_other"
    return None, "none"


def build_rows(utts: list[str], data: dict[str, dict[str, Any]], args: argparse.Namespace,
               rng: random.Random) -> tuple[list[dict[str, Any]], list[tuple[str, str]], dict[str, int]]:
    """Manifest rows for ``utts`` plus the dropped (utt, reason) list and the prompt-reason counts."""
    pools: dict[str, list[str]] = {}
    if args.prompt_mode == "same_spk":
        if args.prompt_pool == "all":
            pool_src = data
            LOG.info("prompt pool = ALL %d utts of %s (the whole Kaldi dir, not only the selected "
                     "subset of %d targets); --prompt_pool subset restricts it", len(data),
                     args.data_dir, len(utts))
        else:
            pool_src = {u: data[u] for u in utts}
            LOG.info("prompt pool = SUBSET: only the %d selected targets", len(utts))
        pools = build_prompt_pools(pool_src, args.min_dur)
    counts = {"window": 0, "any_other": 0, "self": 0}
    dropped: list[tuple[str, str]] = []
    write_ref = common.opencc_available()
    if not write_ref:
        LOG.warning("opencc not importable in this env: text_ref column left EMPTY; "
                    "11_score_mer recomputes seame_normalize(text_raw) in the scoring env")
    rows: list[dict[str, Any]] = []
    for utt in utts:
        row = data[utt]
        if args.prompt_mode == "same_spk":
            p_utt, reason = choose_prompt(utt, row["spk"], pools, data, args.prompt_min, args.prompt_max, rng)
            if p_utt is None:
                if not args.allow_self_prompt:
                    dropped.append((utt, f"{REASON_NO_PROMPT}: speaker {row['spk']} has no other utt in "
                                         f"the prompt pool [{args.min_dur:g}, {PROMPT_HARD_MAX_S:g}] s "
                                         f"(--prompt_pool {args.prompt_pool}); pass --allow_self_prompt "
                                         f"to use the utterance itself"))
                    continue
                p_utt, reason = utt, "self"
        else:
            p_utt, reason = utt, "self"
        counts[reason] += 1
        text_raw = row["text_raw"]
        rows.append({
            "utt": utt,
            "spk": row["spk"],
            "dur": float(row["dur"]),
            "wav": row["wav"],
            "text_raw": text_raw,
            "text_tts": tts_text(text_raw, args.text_mode),
            "text_ref": common.seame_normalize(text_raw) if write_ref else "",
            "prompt_utt": p_utt,
            "prompt_wav": data[p_utt]["wav"],
            "prompt_text": tts_text(data[p_utt]["text_raw"], args.text_mode),
        })
    LOG.info("prompts: %d in window [%.1f, %.1f] s, %d any-other-utt fallback, %d self (%s)",
             counts["window"], args.prompt_min, args.prompt_max, counts["any_other"], counts["self"],
             "prompt == target leak" if counts["self"] else "no leak")
    if args.prompt_mode == "same_spk":
        LOG.info("dropped %d targets whose speaker has no other prompt utt (%s; --allow_self_prompt=%s)",
                 len(dropped), REASON_NO_PROMPT, args.allow_self_prompt)
    return rows, dropped, counts


def write_dropped(path: str, dropped: list[tuple[str, str]]) -> None:
    """One 'utt<TAB>reason' line per dropped target (file always written, possibly empty)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for utt, reason in dropped:
            f.write(f"{utt}\t{reason}\n")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> None:
    common.setup_logging()
    args = parse_args(argv)
    LOG.info("args: %s", vars(args))
    LOG.info("text_mode=%s (%s); opencc t2s available=%s", args.text_mode,
             "tags stripped" if args.text_mode == "strip" else "tags kept, stage-1 SFT convention",
             common.opencc_available())
    rng = random.Random(args.seed)

    data = common.read_kaldi_dir(args.data_dir)
    utts = select_targets(data, args.min_dur, args.max_dur)
    n_available, hours_available = len(utts), utt_hours(utts, data)
    cap_selects_all: bool | None = None
    if args.hours is not None:
        utts, cap_selects_all = subset_by_hours(utts, data, args.hours, rng)
    utts.sort()  # deterministic row order and prompt draws regardless of shuffle
    if args.limit is not None:
        utts = utts[: args.limit]
        LOG.info("--limit %d: %d utts", args.limit, len(utts))
    n_selected, hours_selected = len(utts), utt_hours(utts, data)
    LOG.info("selected %d targets, %.2f h (before prompt assignment)", n_selected, hours_selected)

    rows, dropped, counts = build_rows(utts, data, args, rng)
    common.write_manifest(args.out, rows)
    dropped_path = sidecar_path(args.out, ".dropped.txt")
    write_dropped(dropped_path, dropped)
    n_spk = len({r["spk"] for r in rows})
    hours_written = common.hours(rows)
    mean_cmi = 100.0 * sum(common.text_cmi(r["text_tts"]) for r in rows) / max(1, len(rows))
    LOG.info("wrote %s: %d utts, %d speakers, %.2f h selected (%.2f h available after filtering, "
             "%d dropped for no prompt -> %s), mean text CMI %.2f%%",
             args.out, len(rows), n_spk, hours_written, hours_available, len(dropped), dropped_path,
             mean_cmi)
    summary = {
        "n_utts": len(rows),
        "hours": round(hours_written, 4),
        "n_speakers": n_spk,
        "n_dropped_no_prompt": len(dropped),
        "n_available": n_available,
        "hours_available": round(hours_available, 4),
        "hours_cap_selects_all": cap_selects_all,
        "n_selected": n_selected,
        "hours_selected": round(hours_selected, 4),
        "prompt_counts": counts,
        "mean_text_cmi_pct": round(mean_cmi, 2),
        "out": args.out,
        "dropped_txt": dropped_path,
        "args": vars(args),
    }
    json_path = sidecar_path(args.out, ".json")
    with open(json_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    os.replace(json_path + ".tmp", json_path)
    LOG.info("wrote %s", json_path)


if __name__ == "__main__":
    main()
