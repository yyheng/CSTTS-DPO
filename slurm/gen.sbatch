#!/bin/bash
#SBATCH --job-name=cd_gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=6-00:00:00
# ---------------------------------------------------------------------------
# 10_gen_candidates.py : N sampled candidates per utterance (1 GPU, env $CMI_DPO_ENV_MAIN)
# Shardable: slurm/sb slurm/gen.sbatch --export=ALL,SHARD=i,NSHARDS=n   (n <= 6; each shard writes its own
# OUT/cands.tsv.<shard>, resumable, deterministic per (seed, utt, k)), then once, CPU only (override the
# GPU header so no GPU is held for a seconds-long file merge):
# slurm/sb slurm/gen.sbatch --gres=gpu:0 --cpus-per-task=2 --time=00:30:00 --export=ALL,MERGE=1,OUT=...
#   -> de-duplicated OUT/cands.tsv for 11/12/14.  (run_dpo_round.sh submits this merge itself.)
# Config fingerprint: the script writes OUT/gen_config.json (ckpt, temperature, n_cand, sampling,
# top_p/top_k, seed, text_frontend, text_mode); a resume with a different config ABORTS unless
# FORCE_RESUME=1. DRY_RUN=1 prints the fingerprint + plan (done/pending rows) without loading the model.
# Budget: ~3.3 s per candidate on ~3 s utterances (smoke, incl. warm-up): 80k utts x 4 candidates is
# ~300-350 GPU-h, i.e. 2-3 days per shard with NSHARDS=5..6 (hence the 6-day wall time).
#
# Parameters:
#   ROUND           round dir used for the OUT default       [<pkg>/exp/round1]
#   MANIFEST        manifest TSV from 00                     [<pkg>/data/train_manifest.tsv]
#   OUT             output dir (wav/ tokens/ cands.tsv gen_config.json)  [$ROUND/gen]
#   CKPT            Qwen2LM state_dict (SFT or DPO); "none" = stock CosyVoice2 llm.pt (the script's
#                   --ckpt default is the SFT model, so `none` must be passed explicitly, never omitted)
#                                                            [$CMI_DPO_SFT_CKPT from config/paths.env; none when that is empty]
#   N_CAND          candidates per utterance                 [4]
#   TEMP            sampling temperature                     [1.0]
#   TOP_P / TOP_K   RAS overrides; empty or none = model default  [] / []
#   SAMPLING        Qwen2LM.sampling_ids `sampling`          [25]
#   SEED            base seed                                [0]
#   SHARD / NSHARDS                                          [0 / 1]
#   LIMIT           rows of this shard (0 = all)             [0]
#   SAVE_16K        1 = also write <k>_16k.wav               [0]
#   TEXT_FRONTEND   1 = ttsfrd text_normalize                [0]
#   MAX_RETRIES     sampler retries (--max_retries)          [2]
#   FORCE_RESUME    1 = resume despite a gen_config.json mismatch (fingerprint overwritten) [0]
#   DRY_RUN         1 = fingerprint + plan only, no model load, nothing written  [0]
#   MERGE           1 = only merge OUT/cands.tsv.<shard> into OUT/cands.tsv (no generation) [0]
#   EXTRA           extra CLI flags appended verbatim; none = []   []
# `VAR=none` is the way to pass an empty value through --export (sbatch drops `VAR=`); it is mapped
# to the empty string below for TOP_P TOP_K EXTRA.
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
: "${MANIFEST:=$PKG/data/train_manifest.tsv}"
: "${OUT:=$ROUND/gen}"
: "${CKPT:=${CMI_DPO_SFT_CKPT:-none}}"   # config/paths.env CMI_DPO_SFT_CKPT; empty there = stock llm.pt (none)
: "${N_CAND:=4}"
: "${TEMP:=1.0}"
: "${TOP_P:=}"
: "${TOP_K:=}"
: "${SAMPLING:=25}"
: "${SEED:=0}"
: "${SHARD:=0}"
: "${NSHARDS:=1}"
: "${LIMIT:=0}"
: "${SAVE_16K:=0}"
: "${TEXT_FRONTEND:=0}"
: "${MAX_RETRIES:=2}"
: "${FORCE_RESUME:=0}"
: "${DRY_RUN:=0}"
: "${MERGE:=0}"
: "${EXTRA:=}"
none_is_empty() { local v; for v in "$@"; do if [[ "${!v:-}" == "none" ]]; then printf -v "$v" ''; fi; done; }
none_is_empty TOP_P TOP_K EXTRA

args=(--manifest "$MANIFEST" --out "$OUT" --ckpt "$CKPT" --n_cand "$N_CAND" --temperature "$TEMP"
      --sampling "$SAMPLING" --seed "$SEED" --shard "$SHARD" --nshards "$NSHARDS" --limit "$LIMIT"
      --max_retries "$MAX_RETRIES" --device cuda:0)
if [[ -n "$TOP_P" ]]; then args+=(--top_p "$TOP_P"); fi
if [[ -n "$TOP_K" ]]; then args+=(--top_k "$TOP_K"); fi
if [[ "$SAVE_16K" == "1" ]]; then args+=(--save_16k); fi
if [[ "$TEXT_FRONTEND" == "1" ]]; then args+=(--text_frontend); fi
if [[ "$FORCE_RESUME" == "1" ]]; then args+=(--force_resume); fi
if [[ "$DRY_RUN" == "1" ]]; then args+=(--dry_run); fi

mkdir -p "$OUT"
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-none} gpus=${CUDA_VISIBLE_DEVICES:-?}"
if [[ "$MERGE" == "1" ]]; then
  echo "python -u scripts/10_gen_candidates.py --out $OUT --merge"
  python -u scripts/10_gen_candidates.py --out "$OUT" --merge
  echo 'GEN_DONE'
  exit 0
fi
echo "python -u scripts/10_gen_candidates.py ${args[*]} $EXTRA"
# shellcheck disable=SC2086
python -u scripts/10_gen_candidates.py "${args[@]}" $EXTRA
echo 'GEN_DONE'
