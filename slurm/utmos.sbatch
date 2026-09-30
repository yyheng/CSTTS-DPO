#!/bin/bash
#SBATCH --job-name=cd_utmos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
# ---------------------------------------------------------------------------
# 12_score_utmos.py : UTMOS22-strong naturalness critic, 1 GPU, env $CMI_DPO_ENV_MAIN
# Exactly one of CANDS_TSV / WAV_SCP must be non-empty.
# Failed rows are listed in OUT.failed.txt and skipped (summary line `... failed N`); exit 0 (WARNING)
# while failed / rows in scope <= MAX_FAIL_FRAC, exit 2 above it or with STRICT=1; RESUME=1 retries
# exactly the failed rows (and repairs a torn last line).
#
# Parameters:
#   ROUND        round dir used for the defaults              [<pkg>/exp/round1]
#   CANDS_TSV    cands.tsv from 10                            [$ROUND/gen/cands.tsv]
#   WAV_SCP      Kaldi dir or wav.scp                         []   (set CANDS_TSV=none to use it; `none` = unset)
#   OUT          output TSV (utt cand wav utmos)              [$ROUND/utmos.tsv]
#   LIMIT        first N rows (0 = all)                       [0]
#   CAND_LABEL   cand value for WAV_SCP rows (synth for a 20_synthesize corpus)  [gt]
#   RESUME       1 = append to existing OUT                   [0]
#   MAX_FAIL_FRAC  tolerated failed fraction (--max_fail_frac) [0.01]
#   STRICT       1 = exit 2 on any failed row (--strict)      [0]
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
: "${WAV_SCP:=}"
if [[ "$WAV_SCP" == "none" ]]; then WAV_SCP=; fi
if [[ -n "$WAV_SCP" ]]; then : "${CANDS_TSV:=}"; else : "${CANDS_TSV:=$ROUND/gen/cands.tsv}"; fi
if [[ "$CANDS_TSV" == "none" ]]; then CANDS_TSV=; fi
: "${OUT:=$ROUND/utmos.tsv}"
: "${LIMIT:=0}"
: "${CAND_LABEL:=gt}"
: "${RESUME:=0}"
: "${MAX_FAIL_FRAC:=0.01}"
: "${STRICT:=0}"
: "${EXTRA:=}"
none_is_empty() { local v; for v in "$@"; do if [[ "${!v:-}" == "none" ]]; then printf -v "$v" ''; fi; done; }
none_is_empty EXTRA

args=(--out "$OUT" --limit "$LIMIT" --cand_label "$CAND_LABEL" --max_fail_frac "$MAX_FAIL_FRAC" --device cuda:0)
if [[ -n "$CANDS_TSV" && -n "$WAV_SCP" ]]; then echo "set only one of CANDS_TSV / WAV_SCP" >&2; exit 1; fi
if [[ -n "$CANDS_TSV" ]]; then args+=(--cands_tsv "$CANDS_TSV")
elif [[ -n "$WAV_SCP" ]]; then args+=(--wav_scp "$WAV_SCP")
else echo "one of CANDS_TSV / WAV_SCP is required" >&2; exit 1; fi
if [[ "$RESUME" == "1" ]]; then args+=(--resume); fi
if [[ "$STRICT" == "1" ]]; then args+=(--strict); fi

mkdir -p "$(dirname "$OUT")"
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-none} gpus=${CUDA_VISIBLE_DEVICES:-?}"
echo "python -u scripts/12_score_utmos.py ${args[*]} $EXTRA"
# shellcheck disable=SC2086
python -u scripts/12_score_utmos.py "${args[@]}" $EXTRA
echo 'UTMOS_DONE'
