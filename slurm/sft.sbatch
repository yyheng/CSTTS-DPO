#!/bin/bash
#SBATCH --job-name=cd_sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=5-00:00:00
# ---------------------------------------------------------------------------
# 02_train_sft.py : stage-1 SFT of the CosyVoice2 LLM, torchrun DDP (env $CMI_DPO_ENV_MAIN)
# GPUs: the header requests 4. To use another count override BOTH the allocation and GPUS:
#   slurm/sb slurm/sft.sbatch --gres=gpu:2 --export=ALL,GPUS=2
# (GPUS is auto-derived from SLURM_GPUS_ON_NODE / CUDA_VISIBLE_DEVICES when not given.)
# Batch: BATCH is the micro-batch PER GPU, so the effective batch is BATCH x GPUS x ACCUM (4 x 4 = 16 by
# default). GLOBAL_BATCH=G fixes the effective batch instead (accum = G / (BATCH x GPUS), must divide
# exactly; overrides ACCUM): GLOBAL_BATCH=4 reproduces the paper's "batch size of 4" literally.
# MAX_STEPS / EVAL_EVERY / WARMUP_STEPS count OPTIMIZER steps. Schedule: LR_SCHEDULE=constant_with_warmup
# (paper: linear warm-up, then constant) or linear (warm-up then decay to 0, the old default).
# Side file: EXP/train_state.pt (~4 GB; optimizer + scheduler + counters + geometry of the last checkpoint)
# is written next to every best.pth / epoch_XXX.pth and serves RESUME=<ckpt in EXP>. A mid-epoch checkpoint
# must be resumed with the same GPUS / BATCH / ACCUM (GLOBAL_BATCH) / SEED (refused otherwise); a
# completed-epoch epoch_XXX.pth accepts another GPU count.
#
# Parameters:
#   TRAIN_CACHE    cache path(s): comma list and/or glob      [<pkg>/data/sft_cache/train_manifest.shard*.pt]
#   VALID_CACHE    validation cache(s)                        [<pkg>/data/sft_cache/valid_manifest.shard0.pt]
#   EXP            experiment dir (best.pth, epoch_XXX.pth, train_state.pt)   [<pkg>/exp/sft_seame]
#   GPUS           torchrun --nproc_per_node                  [auto | 4]
#   LR             peak AdamW lr (paper 2e-4)                 [2e-4]
#   LR_SCHEDULE    constant_with_warmup | linear (--lr_schedule)  [constant_with_warmup]
#   BATCH          per-GPU micro-batch (paper 4)              [4]
#   ACCUM          gradient accumulation factor (--accum)     [1]
#   GLOBAL_BATCH   effective batch over all GPUs (--global_batch; overrides ACCUM); empty or none = off  []
#   EPOCHS         max epochs                                 [20]
#   MAX_STEPS      max optimizer steps (paper ~50k)           [50000]
#   WARMUP_STEPS   -1 = 10% of total                          [-1]
#   EVAL_EVERY     validate every N optimizer steps           [1000]
#   PATIENCE       early stopping (0 = off)                   [5]
#   GRAD_CLIP      max grad norm                              [5.0]
#   BF16           1 = bf16 autocast                          [1]
#   SAVE_EVERY_EPOCH 0 = only save epoch_XXX.pth for the last epoch [1]
#   INIT_CKPT      optional state_dict to start from; empty or none = stock llm.pt  []
#   RESUME         checkpoint in EXP to resume from (--resume; weights + optimizer/scheduler/counters
#                  from EXP/train_state.pt when it names that checkpoint); empty or none = fresh  []
#   NUM_WORKERS    DataLoader workers per rank                [2]
#   SEED                                                      [0]
#   EXTRA          extra CLI flags appended verbatim; none = []  []
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
  else GPUS=4; fi
fi
: "${TRAIN_CACHE:=$PKG/data/sft_cache/train_manifest.shard*.pt}"
: "${VALID_CACHE:=$PKG/data/sft_cache/valid_manifest.shard0.pt}"
: "${EXP:=$PKG/exp/sft_seame}"
: "${LR:=2e-4}"
: "${LR_SCHEDULE:=constant_with_warmup}"
: "${BATCH:=4}"
: "${ACCUM:=1}"
: "${GLOBAL_BATCH:=}"
: "${EPOCHS:=20}"
: "${MAX_STEPS:=50000}"
: "${WARMUP_STEPS:=-1}"
: "${EVAL_EVERY:=1000}"
: "${PATIENCE:=5}"
: "${GRAD_CLIP:=5.0}"
: "${BF16:=1}"
: "${SAVE_EVERY_EPOCH:=1}"
: "${INIT_CKPT:=}"
: "${RESUME:=}"
: "${NUM_WORKERS:=2}"
: "${SEED:=0}"
: "${EXTRA:=}"
none_is_empty() { local v; for v in "$@"; do if [[ "${!v:-}" == "none" ]]; then printf -v "$v" ''; fi; done; }
none_is_empty GLOBAL_BATCH INIT_CKPT RESUME EXTRA

args=(--train_cache "$TRAIN_CACHE" --valid_cache "$VALID_CACHE" --exp "$EXP"
      --lr "$LR" --lr_schedule "$LR_SCHEDULE" --batch "$BATCH" --accum "$ACCUM"
      --epochs "$EPOCHS" --max_steps "$MAX_STEPS"
      --warmup_steps "$WARMUP_STEPS" --eval_every "$EVAL_EVERY" --patience "$PATIENCE"
      --grad_clip "$GRAD_CLIP" --num_workers "$NUM_WORKERS" --seed "$SEED")
if [[ -n "$GLOBAL_BATCH" ]]; then args+=(--global_batch "$GLOBAL_BATCH"); fi
if [[ "$BF16" == "1" ]]; then args+=(--bf16); fi
if [[ "$SAVE_EVERY_EPOCH" == "0" ]]; then args+=(--no_save_every_epoch); fi
if [[ -n "$INIT_CKPT" ]]; then args+=(--init_ckpt "$INIT_CKPT"); fi
if [[ -n "$RESUME" ]]; then args+=(--resume "$RESUME"); fi

mkdir -p "$EXP"
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-none} GPUS=$GPUS visible=${CUDA_VISIBLE_DEVICES:-?}"
echo "torchrun --standalone --nproc_per_node=$GPUS scripts/02_train_sft.py ${args[*]} $EXTRA"
# shellcheck disable=SC2086
torchrun --standalone --nproc_per_node="$GPUS" scripts/02_train_sft.py "${args[@]}" $EXTRA
echo 'SFT_DONE'
