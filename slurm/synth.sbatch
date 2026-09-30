#!/bin/bash
#SBATCH --job-name=cd_synth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=6-00:00:00
# ---------------------------------------------------------------------------
# 20_synthesize.py : synthetic Kaldi corpus from a SFT/DPO checkpoint (1 GPU, env $CMI_DPO_ENV_MAIN)
# Shardable: slurm/sb slurm/synth.sbatch --export=ALL,SHARD=i,NSHARDS=n,HOURS=<corpus total> (n <= 6),
# then once, CPU only (override the GPU header; the merge is a seconds-long file concatenation):
# slurm/sb slurm/synth.sbatch --gres=gpu:0 --cpus-per-task=2 --time=00:30:00 --export=ALL,MERGE=1,OUT=...
# HOURS is the CORPUS-level cap (docs/DESIGN.md): the script stops each shard at HOURS/NSHARDS.
# Quality filter: an utterance whose generation hit max_len without EOS (KEEP_TRUNCATED=0) or whose
# wav is longer than MAX_DUR s is re-drawn REJECT_RETRIES times with a fresh seed, then listed in
# failed.txt as "rejected: ..." (no wav / Kaldi line). Rows are processed in a seeded SHUFFLED order
# (speaker-balanced shards and HOURS cap; NO_SHUFFLE=1 keeps manifest order). Config fingerprint
# OUT/synth_config.json: a resume with a different config ABORTS unless FORCE_RESUME=1; DRY_RUN=1
# prints fingerprint + plan (done / pending / beyond the cap) without loading the model.
# The final log line is `DONE shard=i/n rows= done= skipped= failed= rejected= rejected_skipped= redraws= hours_in_shard= capped= out=`
# (rejected_skipped = utts already listed as `rejected:` in this shard's failed file, not re-drawn on resume;
# EXTRA=--retry_rejected re-draws them). MERGE=1 also merges failed.txt.<i> into OUT/failed.txt.
# Budget: ~3.3 s per utterance on ~3 s utts (smoke); 100 h of synthetic audio ~ 90k utts ~ 80 GPU-h
# (14-16 h per shard with NSHARDS=6).
#
# Parameters:
#   MANIFEST      manifest TSV (train transcripts + prompts)  [<pkg>/data/train_manifest.tsv]
#   OUT           corpus dir (wav/ wav.scp text utt2spk utt2dur failed.txt synth_config.json)  [<pkg>/exp/synth_round1]
#   CKPT          Qwen2LM state_dict (SFT or DPO); "none" = stock CosyVoice2 llm.pt (pass it explicitly:
#                 the script's --ckpt default is the SFT model)  [$ROUND/dpo/dpo_best.pth]
#   ROUND         only used for the CKPT default             [<pkg>/exp/round1]
#   SR            output sample rate                         [16000]
#   HOURS         corpus-level cap, split over shards (0 = none) [0]
#   SHARD / NSHARDS                                          [0 / 1]
#   TEMP          sampling temperature                       [1.0]
#   TOP_P / TOP_K RAS overrides; empty or none = model default  [] / []
#   SAMPLING      Qwen2LM.sampling_ids `sampling`            [25]
#   SEED          base seed (also seeds the row shuffle)     [0]
#   LIMIT         rows of this shard (0 = all)               [0]
#   UTT_PREFIX    prefix for output utt ids                  [syn_]
#   TEXT_FRONTEND 1 = ttsfrd text_normalize                  [0]
#   MAX_RETRIES   sampler-error retries (--max_retries)      [2]
#   MAX_DUR       reject utterances longer than (s); inf disables (--max_dur)  [30.0]
#   KEEP_TRUNCATED 1 = keep generations cut by max_len without EOS (--keep_truncated) [0]
#   REJECT_RETRIES fresh-seed re-draws of a rejected utterance (--reject_retries) [1]
#   NO_SHUFFLE    1 = keep manifest order (--no_shuffle)     [0]
#   FORCE_RESUME  1 = resume despite a synth_config.json mismatch (fingerprint overwritten) [0]
#   DRY_RUN       1 = fingerprint + plan only, no model load, nothing written  [0]
#   MERGE         1 = only merge shard files into OUT        [0]
#   EXTRA         extra CLI flags appended verbatim; none = []   []
# `VAR=none` passes an empty value through --export (sbatch drops `VAR=`); mapped to "" for TOP_P TOP_K EXTRA.
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
: "${OUT:=$PKG/exp/synth_round1}"
: "${CKPT:=$ROUND/dpo/dpo_best.pth}"
: "${SR:=16000}"
: "${HOURS:=0}"
: "${SHARD:=0}"
: "${NSHARDS:=1}"
: "${TEMP:=1.0}"
: "${TOP_P:=}"
: "${TOP_K:=}"
: "${SAMPLING:=25}"
: "${SEED:=0}"
: "${LIMIT:=0}"
: "${UTT_PREFIX:=syn_}"
: "${TEXT_FRONTEND:=0}"
: "${MAX_RETRIES:=2}"
: "${MAX_DUR:=30.0}"
: "${KEEP_TRUNCATED:=0}"
: "${REJECT_RETRIES:=1}"
: "${NO_SHUFFLE:=0}"
: "${FORCE_RESUME:=0}"
: "${DRY_RUN:=0}"
: "${MERGE:=0}"
: "${EXTRA:=}"
none_is_empty() { local v; for v in "$@"; do if [[ "${!v:-}" == "none" ]]; then printf -v "$v" ''; fi; done; }
none_is_empty TOP_P TOP_K EXTRA

mkdir -p "$OUT"
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-none} gpus=${CUDA_VISIBLE_DEVICES:-?}"
if [[ "$MERGE" == "1" ]]; then
  echo "python -u scripts/20_synthesize.py --out $OUT --merge"
  python -u scripts/20_synthesize.py --out "$OUT" --merge
  echo 'SYNTH_DONE'
  exit 0
fi
args=(--manifest "$MANIFEST" --out "$OUT" --ckpt "$CKPT" --sr "$SR" --hours "$HOURS"
      --shard "$SHARD" --nshards "$NSHARDS" --temperature "$TEMP" --sampling "$SAMPLING" --seed "$SEED"
      --limit "$LIMIT" --utt_prefix "$UTT_PREFIX" --max_retries "$MAX_RETRIES" --max_dur "$MAX_DUR"
      --reject_retries "$REJECT_RETRIES" --device cuda:0)
if [[ -n "$TOP_P" ]]; then args+=(--top_p "$TOP_P"); fi
if [[ -n "$TOP_K" ]]; then args+=(--top_k "$TOP_K"); fi
if [[ "$TEXT_FRONTEND" == "1" ]]; then args+=(--text_frontend); fi
if [[ "$KEEP_TRUNCATED" == "1" ]]; then args+=(--keep_truncated); fi
if [[ "$NO_SHUFFLE" == "1" ]]; then args+=(--no_shuffle); fi
if [[ "$FORCE_RESUME" == "1" ]]; then args+=(--force_resume); fi
if [[ "$DRY_RUN" == "1" ]]; then args+=(--dry_run); fi
echo "python -u scripts/20_synthesize.py ${args[*]} $EXTRA"
# shellcheck disable=SC2086
python -u scripts/20_synthesize.py "${args[@]}" $EXTRA
echo 'SYNTH_DONE'
