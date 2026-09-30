#!/bin/bash
#SBATCH --job-name=cd_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
# ---------------------------------------------------------------------------
# eval_tts.py : merge the three critic TSVs of one set -> Table-1 metrics + eval.json (CPU, env $CMI_DPO_ENV_MAIN)
#
# Parameters:
#   ROUND        round dir used for the defaults              [<pkg>/exp/round1]
#   SET_DIR      dir with cands.tsv (or wav.scp)              [$ROUND/gen]
#   MER_TSV      11 output                                    [$ROUND/mer.tsv]
#   UTMOS_TSV    12 output                                    [$ROUND/utmos.tsv]
#   CMI_TSV      13 output (candidate rows)                   [$ROUND/cmi_cands.tsv]
#   CMI_GT_TSV   TSV holding the cand=gt rows; empty = CMI_TSV []
#   CAND         only this cand value (e.g. 0); empty = all   []
#   CAND_LABEL   cand value given to wav.scp rows when SET_DIR has no cands.tsv  [gt]
#   UTT_PREFIX   prefix 20_synthesize.py added to the set's utt ids (e.g. syn_); it is stripped
#                before the gt CMI lookup, so dCMI joins for a synth corpus  []
#   SET_NAME     name recorded in eval.json                   [basename of SET_DIR]
#   OUT          eval.json path; empty = SET_DIR/eval.json    []
#   EXTRA        extra CLI flags appended verbatim; none = [] []
# `VAR=none` passes an empty value through --export (sbatch drops `VAR=`); mapped to "" for
# CMI_GT_TSV CAND UTT_PREFIX SET_NAME OUT EXTRA.
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
: "${SET_DIR:=$ROUND/gen}"
: "${MER_TSV:=$ROUND/mer.tsv}"
: "${UTMOS_TSV:=$ROUND/utmos.tsv}"
: "${CMI_TSV:=$ROUND/cmi_cands.tsv}"
: "${CMI_GT_TSV:=}"
: "${CAND:=}"
: "${CAND_LABEL:=gt}"
: "${UTT_PREFIX:=}"
: "${SET_NAME:=}"
: "${OUT:=}"
: "${EXTRA:=}"
none_is_empty() { local v; for v in "$@"; do if [[ "${!v:-}" == "none" ]]; then printf -v "$v" ''; fi; done; }
none_is_empty CMI_GT_TSV CAND UTT_PREFIX SET_NAME OUT EXTRA

args=(--set_dir "$SET_DIR" --mer_tsv "$MER_TSV" --utmos_tsv "$UTMOS_TSV" --cmi_tsv "$CMI_TSV" --cand_label "$CAND_LABEL")
if [[ -n "$UTT_PREFIX" ]]; then args+=(--utt_prefix "$UTT_PREFIX"); fi
if [[ -n "$CMI_GT_TSV" ]]; then args+=(--cmi_gt_tsv "$CMI_GT_TSV"); fi
if [[ -n "$CAND" ]]; then args+=(--cand "$CAND"); fi
if [[ -n "$SET_NAME" ]]; then args+=(--set_name "$SET_NAME"); fi
if [[ -n "$OUT" ]]; then args+=(--out "$OUT"); fi

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "python -u scripts/eval_tts.py ${args[*]} $EXTRA"
# shellcheck disable=SC2086
python -u scripts/eval_tts.py "${args[@]}" $EXTRA
echo 'EVAL_DONE'
