#!/bin/bash
#SBATCH --job-name=cd_pairs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
# ---------------------------------------------------------------------------
# 14_build_pairs.py : critics -> DPO preference pairs (CPU only, env $CMI_DPO_ENV_MAIN)
# If CMI_GT is given (and dCMI is active, NU != 0), the gt rows (cand=gt) are concatenated with the
# candidate rows of CMI into <out dir>/cmi_merged.tsv (headers must match) and that file is passed as --cmi.
# Disabled critics (weight 0): the critic neither scores nor gates (its threshold is skipped unless
# THRESHOLD_DISABLED=1) and its TSV becomes optional: it is passed only when the file exists (values are
# then recorded as information); with NU=0 no gt-CMI merge happens. Ranking pool: candidates with more
# than MAX_CAND_TOKENS speech tokens or cut by max_len without EOS (unless KEEP_TRUNCATED=1) are excluded.
#
# Parameters:
#   ROUND        round dir used for the defaults              [<pkg>/exp/round1]
#   CANDS        cands.tsv                                    [$ROUND/gen/cands.tsv]
#   MER          mer.tsv (required while GAM != 0)            [$ROUND/mer.tsv]
#   UTMOS        utmos.tsv (required while LAM != 0)          [$ROUND/utmos.tsv]
#   CMI          cmi TSV with the candidate rows (required while NU != 0)  [$ROUND/cmi_cands.tsv]
#   CMI_GT       cmi TSV holding the cand=gt rows; empty or none = CMI already contains them  []
#   TOKENS_DIR   tokens/ dir from 10                          [$ROUND/gen/tokens]
#   OUT          output dir (pairs.pt, pairs_summary.json) or a .pt path  [$ROUND/pairs]
#   LAM / GAM / NU   weights of UTMOS / MER / dCMI (0 disables a critic)  [1.0 / 1.0 / 1.0]
#   NORM         global | per_utt                             [global]
#   MER_CLIP     clip MER to [0, MER_CLIP] before normalisation; inf disables (--mer_clip)  [1.0]
#   MAX_MER / MIN_UTMOS / MAX_DCMI  thresholds on the preferred candidate (raw values; inf relaxes)  [0.20 / 2.5 / 0.20]
#   THRESHOLD_DISABLED  1 = also apply the thresholds of weight-0 critics (--threshold_disabled_critics) [0]
#   MAX_CAND_TOKENS  pool filter on speech tokens (--max_cand_tokens; 750 = 30 s)  [750]
#   KEEP_TRUNCATED   1 = keep candidates cut by max_len without EOS in the pool (--keep_truncated) [0]
#   EXTRA        extra CLI flags appended verbatim; none = [] []
# ---------------------------------------------------------------------------
set -euo pipefail
# site configuration: slurm/sb sets CMI_DPO_PKG (package root) and the job re-loads config/paths.env
PKG=${CMI_DPO_PKG:?run through slurm/sb (it sets CMI_DPO_PKG and loads config/paths.env)}
[[ -f "$PKG/config/paths.env" ]] || { echo "missing $PKG/config/paths.env (copy config/paths.env.example)" >&2; exit 1; }
source "$PKG/slurm/load_env.sh"   # CMI_DPO_* already in the environment win over the file
source "${CMI_DPO_CONDA_SH:?set CMI_DPO_CONDA_SH in config/paths.env}"
conda activate "${CMI_DPO_ENV_MAIN:?set CMI_DPO_ENV_MAIN in config/paths.env}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$PKG"
export PYTHONPATH="${CMI_DPO_COSY_ROOT:?set CMI_DPO_COSY_ROOT in config/paths.env}:$CMI_DPO_COSY_ROOT/Matcha-TTS${PYTHONPATH:+:$PYTHONPATH}"

: "${ROUND:=$PKG/exp/round1}"
: "${CANDS:=$ROUND/gen/cands.tsv}"
: "${MER:=$ROUND/mer.tsv}"
: "${UTMOS:=$ROUND/utmos.tsv}"
: "${CMI:=$ROUND/cmi_cands.tsv}"
: "${CMI_GT:=}"
: "${TOKENS_DIR:=$ROUND/gen/tokens}"
: "${OUT:=$ROUND/pairs}"
: "${LAM:=1.0}"
: "${GAM:=1.0}"
: "${NU:=1.0}"
: "${NORM:=global}"
: "${MER_CLIP:=1.0}"
: "${MAX_MER:=0.20}"
: "${MIN_UTMOS:=2.5}"
: "${MAX_DCMI:=0.20}"
: "${THRESHOLD_DISABLED:=0}"
: "${MAX_CAND_TOKENS:=750}"
: "${KEEP_TRUNCATED:=0}"
: "${EXTRA:=}"
none_is_empty() { local v; for v in "$@"; do if [[ "${!v:-}" == "none" ]]; then printf -v "$v" ''; fi; done; }
none_is_empty CMI_GT EXTRA
is_zero() { awk -v x="$1" 'BEGIN { exit !(x + 0 == 0) }'; }   # float-aware "weight == 0"

if [[ "$OUT" == *.pt ]]; then out_dir=$(dirname "$OUT"); else out_dir=$OUT; fi
mkdir -p "$out_dir"
args=(--cands "$CANDS" --tokens_dir "$TOKENS_DIR" --out "$OUT"
      --lam "$LAM" --gam "$GAM" --nu "$NU" --norm "$NORM" --mer_clip "$MER_CLIP"
      --max_mer "$MAX_MER" --min_utmos "$MIN_UTMOS" --max_dcmi "$MAX_DCMI" --max_cand_tokens "$MAX_CAND_TOKENS")
# a critic TSV is mandatory while its weight is non-zero, informational (passed only if present) otherwise
if ! is_zero "$GAM" || [[ -f "$MER" ]]; then args+=(--mer "$MER"); else echo "GAM=0 and $MER missing: MER critic skipped"; fi
if ! is_zero "$LAM" || [[ -f "$UTMOS" ]]; then args+=(--utmos "$UTMOS"); else echo "LAM=0 and $UTMOS missing: UTMOS critic skipped"; fi
cmi_file=$CMI
if ! is_zero "$NU"; then
  if [[ -n "$CMI_GT" ]]; then
    if [[ "$(head -n1 "$CMI")" != "$(head -n1 "$CMI_GT")" ]]; then
      echo "header mismatch between $CMI and $CMI_GT (DUMP_LABELS / TEXT_CMI differs?)" >&2; exit 1
    fi
    cmi_file=$out_dir/cmi_merged.tsv
    { head -n1 "$CMI"; tail -n +2 "$CMI_GT"; tail -n +2 "$CMI"; } > "$cmi_file.tmp"
    mv "$cmi_file.tmp" "$cmi_file"
    echo "merged $(($(wc -l < "$cmi_file") - 1)) cmi rows (gt + candidates) -> $cmi_file"
  fi
  args+=(--cmi "$cmi_file")
elif [[ -f "$CMI" ]]; then
  args+=(--cmi "$CMI")            # NU=0: no gt merge, candidate CMI recorded as information only
else
  echo "NU=0 and $CMI missing: dCMI critic skipped"
fi
if [[ "$THRESHOLD_DISABLED" == "1" ]]; then args+=(--threshold_disabled_critics); fi
if [[ "$KEEP_TRUNCATED" == "1" ]]; then args+=(--keep_truncated); fi
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "python -u scripts/14_build_pairs.py ${args[*]} $EXTRA"
# shellcheck disable=SC2086
python -u scripts/14_build_pairs.py "${args[@]}" $EXTRA
echo 'PAIRS_DONE'
