#!/usr/bin/env python
"""Build DPO preference pairs from the three critics (paper section 3.2.3).

Purpose
-------
Reads the candidate list written by 10_gen_candidates.py (cands.tsv + tokens/<utt>.pt) and the
critic tables mer.tsv (11_score_mer.py), utmos.tsv (12_score_utmos.py) and cmi.tsv
(13_score_cmi.py; contains candidate rows AND 'gt' rows for the ground-truth recordings), then

  0. builds the RANKING POOL per utterance: a candidate is excluded from the pool (and therefore
     can be neither the preferred nor the rejected candidate) when it has more than
     --max_cand_tokens speech tokens (default 750 = 30 s at 25 tok/s) or when its ended_with_eos
     flag in tokens/<utt>.pt is False (cut at max_len = runaway babble; --keep_truncated keeps
     those). Excluded candidates are counted under candidate_counts.cand_excluded_over_max_tokens /
     cand_excluded_truncated in the summary and are also left out of the normalisation ranges.
  1. dCMI(utt, cand) = |cmi(utt, cand) - cmi(utt, 'gt')|
  2. clips MER to [0, --mer_clip] (default 1.0; `inf` disables). Raw MER is unbounded (a
     hallucinated candidate against a 1-token reference scores 10-20), so without the clip a global
     min-max range squeezes every normal candidate to mer_n ~ 0 and MER becomes weightless. The
     number of clipped candidates is logged and stored in the summary (mer_clip.n_clipped). The
     threshold --max_mer and the mer_pos / mer_neg values in pairs.pt use the RAW (unclipped) MER.
  3. min-max normalises every critic to [0, 1] (--norm global: over all pooled candidates,
     default; --norm per_utt: within each utterance); a zero range maps to 0.0
  4. R = lam * utmos_n - gam * mer_n - nu * dcmi_n
  5. per utterance (>= 2 pooled candidates): pos = argmax R, neg = argmin R
  6. drops the pair when R_pos == R_neg (tie) or when the PREFERRED candidate violates a
     quality threshold: mer > --max_mer, utmos < --min_utmos, dcmi > --max_dcmi
     (pass e.g. --max_dcmi inf to disable a threshold)
  7. copies text_ids / prompt_speech / pos / neg token tensors from tokens/<utt>.pt, plus the
     per-candidate ended_with_eos flags as eos_pos / eos_neg (True when the tokens file predates
     the flag) so that 15_train_dpo.py only scores the EOS term for candidates that actually
     sampled EOS (a candidate cut by max_len never did)

Critic ablation (paper Table 1 rows: MER only / MER+UTMOS / all three)
  A critic whose weight is 0 (--lam 0 / --gam 0 / --nu 0) is DISABLED:
    * its TSV flag (--utmos / --mer / --cmi) becomes optional; a missing file or missing rows do
      not drop any candidate (the value is recorded as null when unavailable);
    * its contribution to R is exactly 0;
    * its threshold (--min_utmos / --max_mer / --max_dcmi) is NOT applied unless
      --threshold_disabled_critics is given (then a preferred candidate whose disabled-critic
      value is unavailable is dropped under `pos_critic_missing`);
    * with --nu 0 the gt-CMI merge is skipped: no `missing_gt_cmi` drop and cmi.tsv may lack gt rows.
  With all weights > 0 the behaviour is unchanged: every critic TSV is required and a candidate
  is scored only when all three critics (and the utterance's gt CMI) are present.
  The active critics are logged at start and stored in the summary (`active_critics`).

Outputs (under --out, a directory, or the file itself when --out ends with .pt):
  pairs.pt            list of dicts {utt, text_ids, prompt_speech, pos, neg, eos_pos, eos_neg, r_pos, r_neg,
                      mer_pos, utmos_pos, dcmi_pos, cand_pos, cand_neg, mer_neg, utmos_neg, dcmi_neg}
                      (critic values of a disabled critic may be None)
  pairs_summary.json  config, active critics, counts kept/dropped per reason, candidate-level counts
                      (pool exclusions, missing rows, MER clipping), normalisation ranges and the mean
                      critic values of the preferred vs rejected candidates

Utterance-level drop reasons, in the order they are tested:
  missing_tokens (no readable tokens/<utt>.pt; tested for EVERY utt of cands.tsv, before the critic
  join, since the file is needed for the pool filter), missing_gt_cmi (only when dCMI is active),
  tokens_index_missing (fewer than 2 candidates are indexable in the tokens file although >= 2 have
  all active critics; e.g. cands.tsv lists cand 1 but tokens/<utt>.pt holds one candidate),
  too_few_candidates (< 2 SCORED candidates = present in every active critic AND indexable in the
  tokens file, before the pool filter), too_few_after_pool_filter (< 2 left after the length /
  truncation filter), tie, pos_mer_above_max, pos_utmos_below_min, pos_dcmi_above_max,
  pos_critic_missing, tokens_index_missing again (a selected index that vanished from the tokens
  file between the pool filter and the final load).
  pairs_summary.json candidate_counts ALWAYS carries cand_rows, cand_scored, cand_pooled,
  cand_excluded_over_max_tokens, cand_excluded_truncated, cand_truncated_kept, cand_no_tokens_file,
  cand_index_not_in_tokens, cand_missing_mer / cand_missing_utmos / cand_missing_cmi +
  cand_missing_gt_cmi (active critics) and cand_missing_<critic>_ignored (disabled critics), 0 when
  nothing was counted; other keys (cand_duplicate_rows,
  cand_non_integer_index, n_tokens_mismatch, eos_flag_missing, truncated_pos/neg, ...) appear only
  when non-zero.

The script is fully deterministic (no random numbers are drawn; argmax/argmin ties are broken by
the lowest candidate index).

Flags
-----
  --cands --tokens_dir --out      required
  --mer --utmos --cmi             required only while the corresponding weight is non-zero
  --lam --gam --nu                weights (default 1.0 each; 0 disables the critic, see above)
  --norm global|per_utt           normalisation range (default global)
  --mer_clip F                    clip MER to [0, F] before normalisation (default 1.0; inf disables)
  --max_mer --min_utmos --max_dcmi   thresholds on the preferred candidate (defaults 0.20 / 2.5 / 0.20)
  --threshold_disabled_critics    also apply the thresholds of critics whose weight is 0
  --max_cand_tokens N             pool filter: exclude candidates with more than N tokens (default 750)
  --keep_truncated                pool filter: keep candidates whose ended_with_eos flag is False

Environment: CPU is enough (torch only). No configured
paths are needed; --show_paths prints them (cmi_dpo.paths) and exits.

Example
-------
  python -u scripts/14_build_pairs.py \
      --cands exp/round1/gen/cands.tsv \
      --mer exp/round1/mer.tsv \
      --utmos exp/round1/utmos.tsv \
      --cmi exp/round1/cmi.tsv \
      --tokens_dir exp/round1/gen/tokens \
      --out exp/round1/pairs
  MER-only ablation (paper Table 1 row 1): add `--lam 0 --nu 0` and omit --utmos / --cmi.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmi_dpo import common, paths  # noqa: E402

Key = tuple[str, str]  # (utt, cand) with cand as the string found in the TSV ('0', '1', ..., 'gt')

CRITICS = ('mer', 'utmos', 'dcmi')

# Drop reasons (utterance level) in the order they are tested (tokens_index_missing is tested twice:
# before too_few_candidates and again after the final tokens load, see the module docstring).
REASONS = (
    'missing_tokens',
    'missing_gt_cmi',
    'too_few_candidates',
    'too_few_after_pool_filter',
    'tie',
    'pos_mer_above_max',
    'pos_utmos_below_min',
    'pos_dcmi_above_max',
    'pos_critic_missing',
    'tokens_index_missing',
)


@dataclass
class Cand:
    """One scored candidate with raw critics, normalised critics and the reward R.

    mer / utmos / dcmi are the RAW critic values (None when the critic is disabled and the row is
    unavailable); mer_c is the clipped MER that feeds the normalisation.
    """

    utt: str
    cand: int
    wav: str
    n_tokens: int
    mer: Optional[float]
    utmos: Optional[float]
    dcmi: Optional[float]
    ended_with_eos: Optional[bool] = None  # None = flag absent from the tokens file
    mer_c: Optional[float] = None
    mer_n: float = 0.0
    utmos_n: float = 0.0
    dcmi_n: float = 0.0
    r: float = 0.0


@dataclass
class TokMeta:
    """The small part of tokens/<utt>.pt needed for the pool filter (the tensors are reloaded later)."""

    n_tokens: list[int]
    ended_with_eos: Optional[list[bool]]


class ShowPathsAction(argparse.Action):
    """--show_paths: print cmi_dpo.paths.describe() and exit, before the required flags are checked."""

    def __init__(self, option_strings: list[str], dest: str, **kwargs: Any) -> None:
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS,
                         help=kwargs.get('help', 'print the configured paths (cmi_dpo.paths) and exit'))

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
                 values: Any, option_string: str | None = None) -> None:
        print(paths.describe())
        parser.exit()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split('Environment:')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--cands', required=True, help='cands.tsv from 10_gen_candidates.py (utt cand wav dur n_tokens)')
    p.add_argument('--mer', default=None, help='mer.tsv from 11_score_mer.py (column mer); required unless --gam 0')
    p.add_argument('--utmos', default=None, help='utmos.tsv from 12_score_utmos.py (column utmos); required unless --lam 0')
    p.add_argument('--cmi', default=None,
                   help="cmi.tsv from 13_score_cmi.py (column cmi; candidate rows + cand='gt' rows); required unless --nu 0")
    p.add_argument('--tokens_dir', required=True,
                   help='the tokens/ directory written by 10_gen_candidates.py (<gen_out>/tokens, holding <utt>.pt); '
                        'passing <gen_out> itself also works (<gen_out>/tokens/<utt>.pt is tried as a fallback)')
    p.add_argument('--out', required=True, help='output directory (pairs.pt + pairs_summary.json), or a path ending in .pt')
    p.add_argument('--lam', type=float, default=1.0, help='weight of normalised UTMOS in R (0 disables the critic)')
    p.add_argument('--gam', type=float, default=1.0, help='weight of normalised MER in R (0 disables the critic)')
    p.add_argument('--nu', type=float, default=1.0, help='weight of normalised dCMI in R (0 disables the critic)')
    p.add_argument('--norm', choices=('global', 'per_utt'), default='global',
                   help='min-max normalisation range: over all pooled candidates or within each utterance')
    p.add_argument('--mer_clip', type=float, default=1.0,
                   help='clip each candidate MER to [0, mer_clip] before normalisation (inf disables; default 1.0)')
    p.add_argument('--max_mer', type=float, default=0.20, help='drop the pair if the preferred candidate has (raw) MER above this')
    p.add_argument('--min_utmos', type=float, default=2.5, help='drop the pair if the preferred candidate has UTMOS below this')
    p.add_argument('--max_dcmi', type=float, default=0.20, help='drop the pair if the preferred candidate has dCMI above this')
    p.add_argument('--threshold_disabled_critics', action='store_true',
                   help='also apply the threshold of a critic whose weight is 0 (default: thresholds of disabled critics are skipped)')
    p.add_argument('--max_cand_tokens', type=int, default=750,
                   help='exclude candidates with more speech tokens than this from the ranking pool (750 = 30 s at 25 tok/s)')
    p.add_argument('--keep_truncated', action='store_true',
                   help='keep candidates whose ended_with_eos flag is False (cut at max_len) in the ranking pool')
    p.add_argument('--show_paths', action=ShowPathsAction)
    args = p.parse_args()
    if not (args.mer_clip > 0):
        p.error('--mer_clip must be > 0 (use inf to disable)')
    if args.max_cand_tokens <= 0:
        p.error('--max_cand_tokens must be > 0')
    for weight, flag, path in ((args.gam, '--mer', args.mer), (args.lam, '--utmos', args.utmos), (args.nu, '--cmi', args.cmi)):
        if weight != 0 and path is None:
            p.error(f'{flag} is required while its critic weight is non-zero')
    return args


def _to_float(value: object) -> Optional[float]:
    """Parse a TSV cell as a finite float; None when empty / nan / unparsable."""
    try:
        x = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def load_critic(path: Optional[str], column: str, active: bool) -> dict[Key, float]:
    """Read a critic TSV into {(utt, cand): value}; the last occurrence of a key wins.

    For a disabled critic (active=False) a missing path or file yields an empty table (the values
    are then only informational); for an active critic the file must exist.
    """
    table: dict[Key, float] = {}
    if path is None:
        logging.info('%s: critic disabled and no TSV given -> no values', column)
        return table
    if not os.path.isfile(path):
        if active:
            raise FileNotFoundError(f'{column} TSV not found: {path}')
        logging.warning('%s: critic disabled and TSV missing (%s) -> no values', column, path)
        return table
    n_bad = 0
    n_dup = 0
    for row in common.read_tsv(path):
        key = (str(row['utt']), str(row['cand']))
        val = _to_float(row.get(column))
        if val is None:
            n_bad += 1
            continue
        if key in table:
            n_dup += 1
        table[key] = val
    logging.info('%s: %d rows kept (%d unparsable, %d duplicate keys) from %s', column, len(table), n_bad, n_dup, path)
    return table


def load_tokens(tokens_dir: str, utt: str) -> Optional[dict]:
    """Load <tokens_dir>/<utt>.pt (dict from 10_gen_candidates.py) or None when absent / malformed.

    Falls back to <tokens_dir>/tokens/<utt>.pt so that the generation --out dir is accepted too.
    """
    path = os.path.join(tokens_dir, f'{utt}.pt')
    if not os.path.isfile(path):
        path = os.path.join(tokens_dir, 'tokens', f'{utt}.pt')
        if not os.path.isfile(path):
            return None
    try:
        d = torch.load(path, map_location='cpu')
    except (RuntimeError, EOFError, OSError) as e:
        logging.warning('cannot load %s: %s', path, e)
        return None
    if not isinstance(d, dict) or not all(k in d for k in ('text_ids', 'prompt_speech', 'cands')):
        logging.warning('%s lacks text_ids/prompt_speech/cands', path)
        return None
    return d


def tokens_meta(tok: dict) -> TokMeta:
    """Per-candidate token counts and ended_with_eos flags of a tokens dict (flags None when absent)."""
    n_tokens = [int(torch.as_tensor(c).numel()) for c in tok['cands']]
    flags = tok.get('ended_with_eos')  # absent in tokens files written before the flag existed
    eos: Optional[list[bool]] = None
    if flags is not None and len(flags) == len(n_tokens):
        eos = [bool(f) for f in flags]
    elif flags is not None:
        logging.warning('%s: ended_with_eos has %d entries for %d candidates -> flags ignored',
                        tok.get('utt', '?'), len(flags), len(n_tokens))
    return TokMeta(n_tokens=n_tokens, ended_with_eos=eos)


def seed_counts(active: dict[str, bool]) -> Counter:
    """Counter pre-seeded with the documented candidate_counts keys at 0, so that pairs_summary.json
    always carries them (a plain Counter never inserts a key that was not incremented)."""
    counts: Counter = Counter()
    for k in ('cand_rows', 'cand_scored', 'cand_pooled', 'cand_excluded_over_max_tokens', 'cand_excluded_truncated',
              'cand_truncated_kept', 'cand_no_tokens_file', 'cand_index_not_in_tokens'):
        counts[k] = 0
    for name in ('mer', 'utmos'):
        counts[f'cand_missing_{name}' if active[name] else f'cand_missing_{name}_ignored'] = 0
    if active['dcmi']:  # an active dCMI needs the candidate's cmi row AND the utt's gt row
        counts['cand_missing_cmi'] = 0
        counts['cand_missing_gt_cmi'] = 0
    else:
        counts['cand_missing_dcmi_ignored'] = 0
    return counts


def collect_candidates(
    cands_path: str,
    tokens_dir: str,
    mer: dict[Key, float],
    utmos: dict[Key, float],
    cmi: dict[Key, float],
    active: dict[str, bool],
    max_cand_tokens: int,
    keep_truncated: bool,
) -> tuple[OrderedDict[str, list[Cand]], set[str], Counter, set[str], Counter, Counter]:
    """Join cands.tsv with the critics and apply the pool filter.

    Returns (pooled candidates per utt in cands.tsv order, utts seen in cands.tsv, candidate-level
    counts, utts whose tokens file is unreadable, number of SCORED candidates per utt before the
    pool filter, number of critic-complete candidates per utt whose index is absent from the tokens
    file). The tokens file is loaded (once) for EVERY utt of cands.tsv, before the critic join, so
    `no_tokens` is complete whatever the critic tables hold. A candidate is 'scored' only when every
    ACTIVE critic (and, when dCMI is active, the utterance's gt cmi) is present AND its index exists
    in the tokens file; values of disabled critics are optional (None when absent). The pool filter
    then removes over-length and (unless keep_truncated) truncated candidates.
    """
    per_utt: OrderedDict[str, list[Cand]] = OrderedDict()
    utts_seen: set[str] = set()
    counts: Counter = seed_counts(active)
    n_scored: Counter = Counter()
    n_index_missing: Counter = Counter()
    seen_keys: set[Key] = set()
    no_tokens: set[str] = set()
    meta_cache: dict[str, Optional[TokMeta]] = {}
    for row in common.read_tsv(cands_path):
        counts['cand_rows'] += 1
        utt = str(row['utt'])
        cand_s = str(row['cand'])
        utts_seen.add(utt)
        if utt not in meta_cache:  # every utt seen, independent of the critic join
            tok = load_tokens(tokens_dir, utt)
            meta_cache[utt] = tokens_meta(tok) if tok is not None else None
            if tok is None:
                no_tokens.add(utt)
        key = (utt, cand_s)
        if key in seen_keys:
            counts['cand_duplicate_rows'] += 1
            continue
        seen_keys.add(key)
        try:
            cand_idx = int(cand_s)
        except ValueError:
            counts['cand_non_integer_index'] += 1
            continue
        missing = False
        values: dict[str, Optional[float]] = {}
        for name, table in (('mer', mer), ('utmos', utmos)):
            values[name] = table.get(key)
            if values[name] is None:
                if active[name]:
                    counts[f'cand_missing_{name}'] += 1
                    missing = True
                else:
                    counts[f'cand_missing_{name}_ignored'] += 1
        gt_key = (utt, 'gt')
        if key in cmi and gt_key in cmi:
            values['dcmi'] = abs(cmi[key] - cmi[gt_key])
        else:
            values['dcmi'] = None
            if active['dcmi']:
                if key not in cmi:
                    counts['cand_missing_cmi'] += 1
                if gt_key not in cmi:
                    counts['cand_missing_gt_cmi'] += 1
                missing = True
            else:
                counts['cand_missing_dcmi_ignored'] += 1
        if missing:
            continue
        meta = meta_cache[utt]
        if meta is None:
            counts['cand_no_tokens_file'] += 1
            continue
        n_tok_tsv = _to_float(row.get('n_tokens'))
        if cand_idx < len(meta.n_tokens):
            n_tok = meta.n_tokens[cand_idx]
            eos = meta.ended_with_eos[cand_idx] if meta.ended_with_eos is not None else None
            if n_tok_tsv is not None and int(n_tok_tsv) != n_tok:
                counts['n_tokens_mismatch'] += 1
        else:
            counts['cand_index_not_in_tokens'] += 1
            n_index_missing[utt] += 1
            continue
        n_scored[utt] += 1  # critic-complete AND indexable in the tokens file
        counts['cand_scored'] += 1
        if n_tok > max_cand_tokens:
            counts['cand_excluded_over_max_tokens'] += 1
            continue
        if eos is False and not keep_truncated:
            counts['cand_excluded_truncated'] += 1
            continue
        if eos is False:
            counts['cand_truncated_kept'] += 1
        per_utt.setdefault(utt, []).append(Cand(
            utt=utt,
            cand=cand_idx,
            wav=str(row.get('wav', '')),
            n_tokens=n_tok,
            mer=values['mer'],
            utmos=values['utmos'],
            dcmi=values['dcmi'],
            ended_with_eos=eos,
        ))
        counts['cand_pooled'] += 1
    return per_utt, utts_seen, counts, no_tokens, n_scored, n_index_missing


def clip_mer(per_utt: OrderedDict[str, list[Cand]], mer_clip: float) -> int:
    """Fill mer_c = clip(mer, 0, mer_clip) in place; return the number of candidates whose MER was clipped."""
    n_clipped = 0
    for group in per_utt.values():
        for c in group:
            if c.mer is None:
                c.mer_c = None
                continue
            c.mer_c = min(max(c.mer, 0.0), mer_clip)
            if c.mer_c != c.mer:
                n_clipped += 1
    return n_clipped


def _minmax(values: list[Optional[float]]) -> tuple[float, float]:
    vals = [v for v in values if v is not None]
    return (min(vals), max(vals)) if vals else (0.0, 0.0)


def _norm(x: Optional[float], lo: float, hi: float) -> float:
    """Min-max map to [0, 1]; a (numerically) zero range or a missing value maps to 0.0."""
    if x is None:
        return 0.0
    rng = hi - lo
    if rng < 1e-12:
        return 0.0
    return (x - lo) / rng


def normalise_and_score(
    per_utt: OrderedDict[str, list[Cand]],
    norm: str,
    lam: float,
    gam: float,
    nu: float,
) -> dict[str, dict[str, float]]:
    """Fill mer_n / utmos_n / dcmi_n / r in place; return the global critic ranges (for the summary).

    MER is normalised from the clipped value mer_c. A critic with weight 0 contributes exactly 0
    to R (its normalised value is still filled in when values are available).
    """
    all_c = [c for group in per_utt.values() for c in group]
    ranges = {
        'mer': _minmax([c.mer_c for c in all_c]),
        'utmos': _minmax([c.utmos for c in all_c]),
        'dcmi': _minmax([c.dcmi for c in all_c]),
    }
    for group in per_utt.values():
        if norm == 'per_utt':
            r_mer = _minmax([c.mer_c for c in group])
            r_utmos = _minmax([c.utmos for c in group])
            r_dcmi = _minmax([c.dcmi for c in group])
        else:
            r_mer, r_utmos, r_dcmi = ranges['mer'], ranges['utmos'], ranges['dcmi']
        for c in group:
            c.mer_n = _norm(c.mer_c, *r_mer)
            c.utmos_n = _norm(c.utmos, *r_utmos)
            c.dcmi_n = _norm(c.dcmi, *r_dcmi)
            c.r = 0.0
            if lam != 0:
                c.r += lam * c.utmos_n
            if gam != 0:
                c.r -= gam * c.mer_n
            if nu != 0:
                c.r -= nu * c.dcmi_n
    return {k: {'min': v[0], 'max': v[1]} for k, v in ranges.items()}


def select_pair(group: list[Cand]) -> tuple[Cand, Cand]:
    """(argmax R, argmin R); ties on R are broken towards the lowest candidate index."""
    pos = max(group, key=lambda c: (c.r, -c.cand))
    neg = min(group, key=lambda c: (c.r, c.cand))
    return pos, neg


def threshold_reason(
    pos: Cand,
    max_mer: float,
    min_utmos: float,
    max_dcmi: float,
    active: dict[str, bool],
    threshold_disabled: bool = False,
) -> Optional[str]:
    """First violated quality threshold of the preferred candidate, or None.

    The threshold of a disabled critic is skipped unless threshold_disabled is set; a threshold
    that is applied to an unavailable (None) value yields 'pos_critic_missing'.
    """
    checks = (
        ('mer', pos.mer, lambda v: v > max_mer, 'pos_mer_above_max'),
        ('utmos', pos.utmos, lambda v: v < min_utmos, 'pos_utmos_below_min'),
        ('dcmi', pos.dcmi, lambda v: v > max_dcmi, 'pos_dcmi_above_max'),
    )
    for name, value, violated, reason in checks:
        if not active[name] and not threshold_disabled:
            continue
        if value is None:
            return 'pos_critic_missing'
        if violated(value):
            return reason
    return None


def _as_long(x: object) -> torch.Tensor:
    return torch.as_tensor(x).reshape(-1).to(torch.long).contiguous()


def _opt_float(x: Optional[float]) -> Optional[float]:
    return None if x is None else float(x)


def build_pair(pos: Cand, neg: Cand, tok: dict, counts: Counter) -> Optional[dict]:
    """Assemble one pairs.pt entry; None when a candidate index is missing from the tokens file."""
    cands = tok['cands']
    if pos.cand >= len(cands) or neg.cand >= len(cands):
        return None
    pos_t = _as_long(cands[pos.cand])
    neg_t = _as_long(cands[neg.cand])
    eos_flags = tok.get('ended_with_eos')  # absent in tokens files written before the flag existed
    if eos_flags is None or len(eos_flags) != len(cands):
        counts['eos_flag_missing'] += 1
        eos_pos = eos_neg = True
    else:
        eos_pos, eos_neg = bool(eos_flags[pos.cand]), bool(eos_flags[neg.cand])
        counts['truncated_pos'] += int(not eos_pos)
        counts['truncated_neg'] += int(not eos_neg)
    return {
        'utt': pos.utt,
        'text_ids': _as_long(tok['text_ids']),
        'prompt_speech': _as_long(tok['prompt_speech']),
        'pos': pos_t,
        'neg': neg_t,
        'eos_pos': eos_pos,
        'eos_neg': eos_neg,
        'r_pos': float(pos.r),
        'r_neg': float(neg.r),
        'mer_pos': _opt_float(pos.mer),
        'utmos_pos': _opt_float(pos.utmos),
        'dcmi_pos': _opt_float(pos.dcmi),
        'cand_pos': int(pos.cand),
        'cand_neg': int(neg.cand),
        'mer_neg': _opt_float(neg.mer),
        'utmos_neg': _opt_float(neg.utmos),
        'dcmi_neg': _opt_float(neg.dcmi),
    }


def _mean(values: list[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def critic_table(pairs: list[dict]) -> dict[str, dict[str, Optional[float]]]:
    """Mean raw critic values and R of the preferred vs rejected candidates (None values ignored)."""
    return {
        'pos': {
            'mer': _mean([p['mer_pos'] for p in pairs]),
            'utmos': _mean([p['utmos_pos'] for p in pairs]),
            'dcmi': _mean([p['dcmi_pos'] for p in pairs]),
            'R': _mean([p['r_pos'] for p in pairs]),
        },
        'neg': {
            'mer': _mean([p['mer_neg'] for p in pairs]),
            'utmos': _mean([p['utmos_neg'] for p in pairs]),
            'dcmi': _mean([p['dcmi_neg'] for p in pairs]),
            'R': _mean([p['r_neg'] for p in pairs]),
        },
    }


def resolve_out(out: str) -> tuple[str, str]:
    """(pairs.pt path, pairs_summary.json path) from --out (directory or .pt file)."""
    if out.endswith('.pt'):
        out_dir = os.path.dirname(os.path.abspath(out)) or '.'
        pt_path = os.path.abspath(out)
    else:
        out_dir = os.path.abspath(out)
        pt_path = os.path.join(out_dir, 'pairs.pt')
    os.makedirs(out_dir, exist_ok=True)
    return pt_path, os.path.join(out_dir, 'pairs_summary.json')


def _json_float(x: float) -> object:
    """inf is not valid JSON; store it as the string 'inf'."""
    return x if math.isfinite(x) else str(x)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    pt_path, summary_path = resolve_out(args.out)

    active = {'mer': args.gam != 0, 'utmos': args.lam != 0, 'dcmi': args.nu != 0}
    weights = {'mer': args.gam, 'utmos': args.lam, 'dcmi': args.nu}
    logging.info('active critics: %s | disabled: %s | thresholds of disabled critics: %s',
                 ', '.join(f'{k} (weight {weights[k]})' for k in CRITICS if active[k]) or 'none',
                 ', '.join(k for k in CRITICS if not active[k]) or 'none',
                 'APPLIED (--threshold_disabled_critics)' if args.threshold_disabled_critics else 'skipped')
    if not any(active.values()):
        logging.warning('all critic weights are 0: R is 0 for every candidate -> every utterance will be a tie')

    mer = load_critic(args.mer, 'mer', active['mer'])
    utmos = load_critic(args.utmos, 'utmos', active['utmos'])
    cmi = load_critic(args.cmi, 'cmi', active['dcmi'])
    n_gt = sum(1 for (_, c) in cmi if c == 'gt')
    if active['dcmi']:
        logging.info('cmi.tsv holds %d gt rows', n_gt)
    else:
        logging.info('dCMI disabled (--nu 0): gt-CMI merge skipped (%d gt rows present, informational only)', n_gt)

    per_utt, utts_seen, counts, no_tokens, n_scored, n_index_missing = collect_candidates(
        args.cands, args.tokens_dir, mer, utmos, cmi, active, args.max_cand_tokens, args.keep_truncated)
    logging.info('cands.tsv: %d rows, %d utts, %d scored candidates, %d in the ranking pool over %d utts '
                 '(excluded: %d over %d tokens, %d truncated%s)',
                 counts['cand_rows'], len(utts_seen), counts['cand_scored'], counts['cand_pooled'], len(per_utt),
                 counts['cand_excluded_over_max_tokens'], args.max_cand_tokens, counts['cand_excluded_truncated'],
                 f'; {counts["cand_truncated_kept"]} truncated kept by --keep_truncated' if args.keep_truncated else '')

    n_clipped = clip_mer(per_utt, args.mer_clip)
    logging.info('MER clipped to [0, %s]: %d / %d pooled candidates clipped', args.mer_clip, n_clipped, counts['cand_pooled'])

    ranges = normalise_and_score(per_utt, args.norm, args.lam, args.gam, args.nu)
    logging.info('critic ranges (%s; mer = clipped): %s', args.norm, json.dumps(ranges))

    drops: Counter = Counter()
    pairs: list[dict] = []
    for utt in sorted(utts_seen):
        if utt in no_tokens:
            drops['missing_tokens'] += 1
            continue
        if active['dcmi'] and (utt, 'gt') not in cmi:
            drops['missing_gt_cmi'] += 1
            continue
        if n_scored[utt] < 2:
            # enough critic-complete candidates, but the tokens file does not hold them -> tokens problem
            if n_scored[utt] + n_index_missing[utt] >= 2:
                drops['tokens_index_missing'] += 1
            else:
                drops['too_few_candidates'] += 1
            continue
        group = per_utt.get(utt, [])
        if len(group) < 2:
            drops['too_few_after_pool_filter'] += 1
            continue
        pos, neg = select_pair(group)
        if not pos.r > neg.r:
            drops['tie'] += 1
            continue
        reason = threshold_reason(pos, args.max_mer, args.min_utmos, args.max_dcmi, active,
                                  args.threshold_disabled_critics)
        if reason is not None:
            drops[reason] += 1
            continue
        tok = load_tokens(args.tokens_dir, utt)
        if tok is None:  # readable during the pool filter, gone now
            drops['missing_tokens'] += 1
            continue
        pair = build_pair(pos, neg, tok, counts)
        if pair is None:
            drops['tokens_index_missing'] += 1
            continue
        pairs.append(pair)

    torch.save(pairs, pt_path)
    table = critic_table(pairs)
    summary = {
        'config': {
            'cands': os.path.abspath(args.cands),
            'mer': os.path.abspath(args.mer) if args.mer else None,
            'utmos': os.path.abspath(args.utmos) if args.utmos else None,
            'cmi': os.path.abspath(args.cmi) if args.cmi else None,
            'tokens_dir': os.path.abspath(args.tokens_dir),
            'lam': args.lam, 'gam': args.gam, 'nu': args.nu, 'norm': args.norm,
            'mer_clip': _json_float(args.mer_clip),
            'max_mer': _json_float(args.max_mer), 'min_utmos': _json_float(args.min_utmos),
            'max_dcmi': _json_float(args.max_dcmi),
            'threshold_disabled_critics': bool(args.threshold_disabled_critics),
            'max_cand_tokens': args.max_cand_tokens,
            'keep_truncated': bool(args.keep_truncated),
        },
        'active_critics': {k: bool(active[k]) for k in CRITICS},
        'n_utts': len(utts_seen),
        'n_pairs': len(pairs),
        'dropped_utts': {r: drops.get(r, 0) for r in REASONS},
        'candidate_counts': dict(sorted(counts.items())),
        'pool_filter': {
            'max_cand_tokens': args.max_cand_tokens,
            'keep_truncated': bool(args.keep_truncated),
            'n_scored': counts['cand_scored'],
            'n_pooled': counts['cand_pooled'],
            'n_excluded_over_max_tokens': counts['cand_excluded_over_max_tokens'],
            'n_excluded_truncated': counts['cand_excluded_truncated'],
        },
        'mer_clip': {'value': _json_float(args.mer_clip), 'n_clipped': n_clipped, 'n_pooled': counts['cand_pooled']},
        'critic_ranges_global': ranges,
        'mean_critics': table,
        'pairs_pt': pt_path,
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info('kept %d pairs / %d utts; drops: %s', len(pairs), len(utts_seen),
                 ', '.join(f'{r}={drops.get(r, 0)}' for r in REASONS))
    logging.info('%-6s %10s %10s %10s %10s', 'side', 'mer', 'utmos', 'dcmi', 'R')
    for side in ('pos', 'neg'):
        t = table[side]
        logging.info('%-6s %10s %10s %10s %10s', side,
                     *[('%.4f' % t[k]) if t[k] is not None else 'n/a' for k in ('mer', 'utmos', 'dcmi', 'R')])
    logging.info('wrote %s and %s', pt_path, summary_path)
    if not pairs and drops.get('missing_tokens', 0) > 0:
        logging.error('0 pairs kept and every remaining utterance (%d) lacked its tokens file: '
                      'neither %s/<utt>.pt nor %s/tokens/<utt>.pt exists -- pass the tokens/ dir '
                      'written by 10_gen_candidates.py as --tokens_dir',
                      drops['missing_tokens'], args.tokens_dir, args.tokens_dir)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
