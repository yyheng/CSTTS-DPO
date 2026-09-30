#!/bin/bash
#SBATCH --job-name=cd_dpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00
# ---------------------------------------------------------------------------
# 15_train_dpo.py : multi-critic DPO of the CosyVoice2 LLM, torchrun DDP (env $CMI_DPO_ENV_MAIN)
# GPUs: the header requests 2. For another count override both the allocation and GPUS:
#   slurm/sb slurm/dpo.sbatch --gres=gpu:4 --export=ALL,GPUS=4
# (GPUS is auto-derived from SLURM_GPUS_ON_NODE / CUDA_VISIBLE_DEVICES when not given.)
# Model selection: VAL_FRAC of the pairs is held out (seeded; >= 20 pairs needed, else 0) and written to
# EXP/val_pairs.txt; dpo_best.pth = lowest validation loss (else lowest epoch training loss).
# Precision: BF16 applies to the POLICY forward only; the reference log-probs (cache and on-the-fly)
# are always float32 (so with BF16=1 the step-0 margin is bf16 rounding noise, not exactly 0; the log's
# `step-0 check` line shows it). Side files: EXP/train_state.pt (optimizer + counters + geometry of the last
# checkpoint), EXP/ref_cache.pt, EXP/metrics.jsonl (val_loss/val_margin/val_acc per epoch when a validation set exists).
# RESUME geometry: a resume is refused when SEED / VAL_FRAC / MAX_PAIR_TOKENS / the pair count differ from the
# run that wrote train_state.pt, and when GPUS / BATCH / ACCUM differ for a mid-epoch checkpoint (dpo_step_*);
# another GPU count is fine from a completed-epoch dpo_epoch_XXX.pth.
#
# Parameters:
#   ROUND        round dir used for the defaults              [<pkg>/exp/round1]
#   PAIRS        pairs.pt from 14                             [$ROUND/pairs/pairs.pt]
#   EXP          experiment dir (dpo_best.pth, dpo_epoch_XXX.pth, metrics.jsonl, val_pairs.txt)  [$ROUND/dpo]
#   INIT_CKPT    policy/reference init: a state_dict path; empty = script default (stage-1 SFT
#                epoch_020.pth); `none` = the STOCK CosyVoice2 llm.pt (--init_ckpt none)  []
#   RESUME       dpo_step_XXXXXX.pth / dpo_epoch_XXX.pth in EXP to resume from (policy + optimizer);
#                empty or none = fresh run  []
#   GPUS         torchrun --nproc_per_node                    [auto | 2]
#   BETA         DPO beta                                     [0.1]
#   LR           AdamW lr                                     [1e-6]
#   BATCH        pairs per GPU per micro-batch                [4]
#   ACCUM        gradient accumulation                        [1]
#   EPOCHS                                                    [2]
#   GRAD_CLIP                                                 [1.0]
#   BF16         1 = bf16 autocast for the policy forward (reference always float32)  [1]
#   SFT_WEIGHT   SFT regulariser weight                       [0]
#   LENGTH_NORM  1 = length-normalised log-probs              [0]
#   CACHE_REF    1 = pre-compute reference log-probs (float32) [1]
#   VAL_FRAC     held-out fraction for model selection (--val_frac; 0 = none)  [0.05]
#   MAX_PAIR_TOKENS  skip pairs whose pos/neg has more speech tokens (--max_pair_tokens; <= 0 off)  [750]
#   LOG_EVERY / SAVE_EVERY / MAX_STEPS                        [10 / 0 / 0]
#   SEED                                                      [0]
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

if [[ -z "${GPUS:-}" ]]; then
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then GPUS=$SLURM_GPUS_ON_NODE
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then GPUS=$(( $(tr -cd , <<<"$CUDA_VISIBLE_DEVICES" | wc -c) + 1 ))
  else GPUS=2; fi
fi
: "${ROUND:=$PKG/exp/round1}"
: "${PAIRS:=$ROUND/pairs/pairs.pt}"
: "${EXP:=$ROUND/dpo}"
: "${INIT_CKPT:=}"
: "${RESUME:=}"
: "${BETA:=0.1}"
: "${LR:=1e-6}"
: "${BATCH:=4}"
: "${ACCUM:=1}"
: "${EPOCHS:=2}"
: "${GRAD_CLIP:=1.0}"
: "${BF16:=1}"
: "${SFT_WEIGHT:=0}"
: "${LENGTH_NORM:=0}"
: "${CACHE_REF:=1}"
: "${VAL_FRAC:=0.05}"
: "${MAX_PAIR_TOKENS:=750}"
: "${LOG_EVERY:=10}"
: "${SAVE_EVERY:=0}"
: "${MAX_STEPS:=0}"
: "${SEED:=0}"
: "${EXTRA:=}"
none_is_empty() { local v; for v in "$@"; do if [[ "${!v:-}" == "none" ]]; then printf -v "$v" ''; fi; done; }
none_is_empty RESUME EXTRA     # NOT INIT_CKPT: `none` there means "stock llm.pt" and is passed through

args=(--pairs "$PAIRS" --exp "$EXP" --beta "$BETA" --lr "$LR" --batch "$BATCH" --accum "$ACCUM"
      --epochs "$EPOCHS" --grad_clip "$GRAD_CLIP" --sft_weight "$SFT_WEIGHT"
      --val_frac "$VAL_FRAC" --max_pair_tokens "$MAX_PAIR_TOKENS"
      --log_every "$LOG_EVERY" --save_every "$SAVE_EVERY" --max_steps "$MAX_STEPS" --seed "$SEED")
if [[ -n "$INIT_CKPT" ]]; then args+=(--init_ckpt "$INIT_CKPT"); fi
if [[ -n "$RESUME" ]]; then args+=(--resume "$RESUME"); fi
if [[ "$BF16" == "1" ]]; then args+=(--bf16); fi
if [[ "$LENGTH_NORM" == "1" ]]; then args+=(--length_norm); fi
if [[ "$CACHE_REF" == "1" ]]; then args+=(--cache_ref); else args+=(--no-cache_ref); fi

mkdir -p "$EXP"
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-none} GPUS=$GPUS visible=${CUDA_VISIBLE_DEVICES:-?}"
echo "torchrun --standalone --nproc_per_node=$GPUS scripts/15_train_dpo.py ${args[*]} $EXTRA"
# shellcheck disable=SC2086
torchrun --standalone --nproc_per_node="$GPUS" scripts/15_train_dpo.py "${args[@]}" $EXTRA
echo 'DPO_DONE'
