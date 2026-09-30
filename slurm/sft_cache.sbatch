#!/bin/bash
#SBATCH --job-name=cd_sftcache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
# ---------------------------------------------------------------------------
# 01_build_sft_cache.py : manifest -> {text_token, speech_token} cache .pt (1 GPU, env $CMI_DPO_ENV_MAIN)
# Shardable: submit one job per shard, e.g.
#   for i in 0 1 2 3; do slurm/sb slurm/sft_cache.sbatch --export=ALL,SHARD=$i,NSHARDS=4; done   (<= 6 at once)
#
# Parameters:
#   MANIFEST        manifest TSV from 00                     [<pkg>/data/train_manifest.tsv]
#   OUT             output .pt (+ <stem>.json)               [<pkg>/data/sft_cache/<manifest stem>.shard${SHARD}.pt]
#   SHARD / NSHARDS shard index / count (round-robin)        [0 / 1]
#   LIMIT           first N rows of the shard (0 = all)      [0]
#   MAX_DUR         skip audio longer than (s)               [30]
#   MIN_TEXT_TOKENS / MAX_TEXT_TOKENS                        [1 / 200]
#   TEXT_FRONTEND   1 = apply ttsfrd text_normalize          [0]
#   TEXT_COL        manifest column to tokenise (text_tts | text_raw) [text_tts]
#   OVERWRITE       1 = rebuild an existing OUT / discard <OUT>.partial [0]
#   (resumable: the script checkpoints to <OUT>.partial every --log_every rows and
#    continues from it when resubmitted with the same parameters)
#   EXTRA           extra CLI flags appended verbatim; none = []  []
# `VAR=none` passes an empty value through --export (sbatch drops `VAR=`); mapped to "" for EXTRA.
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

: "${MANIFEST:=$PKG/data/train_manifest.tsv}"
: "${SHARD:=0}"
: "${NSHARDS:=1}"
stem=$(basename "$MANIFEST" .tsv)
: "${OUT:=$PKG/data/sft_cache/${stem}.shard${SHARD}.pt}"
: "${LIMIT:=0}"
: "${MAX_DUR:=30}"
: "${MIN_TEXT_TOKENS:=1}"
: "${MAX_TEXT_TOKENS:=200}"
: "${TEXT_FRONTEND:=0}"
: "${TEXT_COL:=text_tts}"
: "${OVERWRITE:=0}"
: "${EXTRA:=}"
none_is_empty() { local v; for v in "$@"; do if [[ "${!v:-}" == "none" ]]; then printf -v "$v" ''; fi; done; }
none_is_empty EXTRA

args=(--manifest "$MANIFEST" --out "$OUT" --shard "$SHARD" --nshards "$NSHARDS" --limit "$LIMIT"
      --max_dur "$MAX_DUR" --min_text_tokens "$MIN_TEXT_TOKENS" --max_text_tokens "$MAX_TEXT_TOKENS"
      --text_col "$TEXT_COL" --device cuda:0)
if [[ "$TEXT_FRONTEND" == "1" ]]; then args+=(--text_frontend); fi
if [[ "$OVERWRITE" == "1" ]]; then args+=(--overwrite); fi

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-none} gpus=${CUDA_VISIBLE_DEVICES:-?}"
echo "python -u scripts/01_build_sft_cache.py ${args[*]} $EXTRA"
# shellcheck disable=SC2086
python -u scripts/01_build_sft_cache.py "${args[@]}" $EXTRA
echo 'SFT_CACHE_DONE'
