#!/bin/bash
# ---------------------------------------------------------------------------
# smoke.sh : sequential end-to-end smoke test of the whole cmi_dpo pipeline on 4 utterances of
# SEAME valid. Every step is a BLOCKING srun on a compute node (<= 1 GPU, 15 min), so nothing
# python runs on the login node. Aborts at the first failing step and names it.
#
#   00 prep (CPU)  -> 01 sft cache (GPU)  -> [02 sft, 2 steps, only with SMOKE_SFT=1]
#   -> 10 gen (2 cands) -> 11 mer -> 12 utmos -> 13 cmi (gt + cands) -> 14 pairs (no thresholds)
#   -> 15 dpo (torchrun, 1 GPU, 2 steps) -> 20 synth (2 utts, dpo ckpt) -> eval_tts
#
# Site configuration: PKG is derived from this file's location and config/paths.env is loaded
# through slurm/load_env.sh (CMI_DPO_* variables already set in the shell win over the file; the
# compute-node shell re-loads it the same way). The srun options come from CMI_DPO_SLURM_PARTITION /
# _EXCLUDE (if set) / _ACCOUNT (if set) / _EXTRA; each step activates the conda env of its role
# (main / asr / lal = CMI_DPO_ENV_MAIN / _ASR / _LAL) via CMI_DPO_CONDA_SH and puts CMI_DPO_COSY_ROOT
# (+ Matcha-TTS) on PYTHONPATH.
#
# Parameters: SMOKE_OUT [<pkg>/smoke_out] (wiped first unless SMOKE_KEEP=1), SMOKE_DATA
# [$CMI_DPO_DATA_ROOT/valid], SMOKE_SFT [0], SRUN_TIME [00:15:00].
# Usage:  bash slurm/smoke.sh        (from the repo root; any cwd works)
# ---------------------------------------------------------------------------
set -uo pipefail
PKG=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_FILE=$PKG/config/paths.env
[[ -f "$ENV_FILE" ]] || { echo "$ENV_FILE not found; copy config/paths.env.example to it and fill in your paths" >&2; exit 1; }
# shellcheck disable=SC1091
source "$PKG/slurm/load_env.sh"   # CMI_DPO_* already in the environment win over the file
export CMI_DPO_PKG=$PKG
: "${CMI_DPO_SLURM_PARTITION:?set CMI_DPO_SLURM_PARTITION in $ENV_FILE}"
: "${CMI_DPO_CONDA_SH:?set CMI_DPO_CONDA_SH in $ENV_FILE}"
: "${CMI_DPO_COSY_ROOT:?set CMI_DPO_COSY_ROOT in $ENV_FILE}"
: "${CMI_DPO_DATA_ROOT:?set CMI_DPO_DATA_ROOT in $ENV_FILE (SMOKE_DATA default = \$CMI_DPO_DATA_ROOT/valid)}"
COSY=$CMI_DPO_COSY_ROOT
: "${SMOKE_OUT:=$PKG/smoke_out}"
: "${SMOKE_DATA:=$CMI_DPO_DATA_ROOT/valid}"
: "${SMOKE_SFT:=0}"
: "${SMOKE_KEEP:=0}"
: "${SRUN_TIME:=00:15:00}"
S=$SMOKE_OUT
LOGDIR=$S/logs

if [[ "$SMOKE_KEEP" != "1" && -d "$S" ]]; then echo "wiping $S"; rm -rf "$S"; fi
mkdir -p "$LOGDIR"

SRUN=(srun --partition="$CMI_DPO_SLURM_PARTITION")
[[ -n "${CMI_DPO_SLURM_EXCLUDE:-}" ]] && SRUN+=(--exclude="$CMI_DPO_SLURM_EXCLUDE")
[[ -n "${CMI_DPO_SLURM_ACCOUNT:-}" ]] && SRUN+=(--account="$CMI_DPO_SLURM_ACCOUNT")
# CMI_DPO_SLURM_EXTRA holds verbatim srun/sbatch options (word-split on purpose)
# shellcheck disable=SC2206
[[ -n "${CMI_DPO_SLURM_EXTRA:-}" ]] && SRUN+=(${CMI_DPO_SLURM_EXTRA})
SRUN+=(--nodes=1 --ntasks=1 --cpus-per-task=4 --time="$SRUN_TIME")

env_of_role() {  # main | asr | lal -> conda env name from config/paths.env
  case "$1" in
    main) echo "${CMI_DPO_ENV_MAIN:?set CMI_DPO_ENV_MAIN in $ENV_FILE}" ;;
    asr)  echo "${CMI_DPO_ENV_ASR:?set CMI_DPO_ENV_ASR in $ENV_FILE}" ;;
    lal)  echo "${CMI_DPO_ENV_LAL:?set CMI_DPO_ENV_LAL in $ENV_FILE}" ;;
    *) echo "unknown env role '$1' (main | asr | lal)" >&2; return 1 ;;
  esac
}

# step <name> <role> <ngpu> <command string>   (command runs inside bash -lc on the compute node;
# role = main | asr | lal selects the conda env CMI_DPO_ENV_MAIN / _ASR / _LAL). The prelude string is
# re-parsed by the remote shell, so PKG and the env name are inserted shell-quoted (printf %q) and the
# site paths are read from the CMI_DPO_* variables of the remote environment (srun forwards them and
# load_env.sh fills in whatever is missing from config/paths.env).
step() {
  local name=$1 role=$2 ngpu=$3 cmd=$4
  local env gres=() q_pkg q_env
  env=$(env_of_role "$role") || exit 1
  (( ngpu > 0 )) && gres=(--gres=gpu:"$ngpu")
  printf -v q_pkg %q "$PKG"
  printf -v q_env %q "$env"
  local prelude="set -euo pipefail; export CMI_DPO_PKG=$q_pkg; source $q_pkg/slurm/load_env.sh; \
source \"\$CMI_DPO_CONDA_SH\"; conda activate $q_env; \
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1; cd $q_pkg; \
export PYTHONPATH=\"\$CMI_DPO_COSY_ROOT:\$CMI_DPO_COSY_ROOT/Matcha-TTS\${PYTHONPATH:+:\$PYTHONPATH}\"; echo \"[host \$(hostname) env $q_env]\""
  echo "==================== [$name] role=$role env=$env gpu=$ngpu  ($(date))"
  echo "$cmd"
  local t0=$SECONDS
  if "${SRUN[@]}" "${gres[@]}" --job-name="smoke_$name" bash -lc "$prelude; $cmd" 2>&1 | tee "$LOGDIR/$name.log"; then
    echo "-------------------- [$name] OK in $((SECONDS - t0)) s"
  else
    echo "SMOKE FAILED at step $name (see $LOGDIR/$name.log)"
    exit 1
  fi
}

# the step commands are re-parsed by the remote shell too: shell-quoted copies of the paths they use
M=$S/manifest.tsv
printf -v QS %q "$S"
printf -v QM %q "$M"
printf -v QD %q "$SMOKE_DATA"

step prep main 0 \
  "python -u scripts/00_prep_manifest.py --data_dir $QD --out $QM --limit 4 --seed 0"

step sft_cache main 1 \
  "python -u scripts/01_build_sft_cache.py --manifest $QM --out $QS/sft_cache/valid.pt --limit 4 --overwrite"

if [[ "$SMOKE_SFT" == "1" ]]; then
  step sft main 1 \
    "torchrun --standalone --nproc_per_node=1 scripts/02_train_sft.py --train_cache $QS/sft_cache/valid.pt --valid_cache $QS/sft_cache/valid.pt --exp $QS/sft --batch 1 --epochs 1 --max_steps 2 --eval_every 2 --log_every 1 --num_workers 0"
fi

step gen main 1 \
  "python -u scripts/10_gen_candidates.py --manifest $QM --out $QS/gen --n_cand 2 --temperature 1.0 --seed 0"

step mer asr 1 \
  "python -u scripts/11_score_mer.py --cands_tsv $QS/gen/cands.tsv --manifest $QM --out $QS/mer.tsv --batch_size 4 --jiwer_check"

step utmos main 1 \
  "python -u scripts/12_score_utmos.py --cands_tsv $QS/gen/cands.tsv --out $QS/utmos.tsv"

step cmi lal 1 \
  "python -u scripts/13_score_cmi.py --cands_tsv $QS/gen/cands.tsv --manifest $QM --out $QS/cmi.tsv --batch 4 --print_summary"

step pairs main 0 \
  "python -u scripts/14_build_pairs.py --cands $QS/gen/cands.tsv --mer $QS/mer.tsv --utmos $QS/utmos.tsv --cmi $QS/cmi.tsv --tokens_dir $QS/gen/tokens --out $QS/pairs --max_mer 10 --min_utmos 0 --max_dcmi 10 && cat $QS/pairs/pairs_summary.json"

step dpo main 1 \
  "torchrun --standalone --nproc_per_node=1 scripts/15_train_dpo.py --pairs $QS/pairs/pairs.pt --exp $QS/dpo --max_steps 2 --batch 1 --epochs 1 --log_every 1 && ls -la $QS/dpo"

step synth main 1 \
  "python -u scripts/20_synthesize.py --manifest $QM --ckpt $QS/dpo/dpo_best.pth --out $QS/synth --limit 2 && cat $QS/synth/utt2dur"

step eval main 0 \
  "python -u scripts/eval_tts.py --set_dir $QS/gen --mer_tsv $QS/mer.tsv --utmos_tsv $QS/utmos.tsv --cmi_tsv $QS/cmi.tsv --set_name smoke --out $QS/eval.json && cat $QS/eval.json"

echo "SMOKE_DONE all steps passed; outputs in $S"
