#!/bin/bash
#SBATCH --job-name=cd_mer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
# ---------------------------------------------------------------------------
# 11_score_mer.py : MER critic (SEAME-FT whisper-large-v3), 1 GPU, env $CMI_DPO_ENV_ASR
# Exactly one of CANDS_TSV / WAV_SCP must be non-empty.
# Failed rows (unreadable wav, ...) are listed in OUT.failed.txt and skipped; exit status 0 (with a
# WARNING) while failed / rows in scope <= MAX_FAIL_FRAC, exit 2 above it or with STRICT=1 (the
# afterok chain then stops); re-run with RESUME=1 to retry exactly the failed rows (RESUME also
# repairs a last line torn by a hard kill). Budget: ~320k candidates (80k utts x 4) ~ 10 h estimated.
#
# Parameters:
#   ROUND        round dir used for the defaults              [<pkg>/exp/round1]
#   CANDS_TSV    cands.tsv from 10 (needs MANIFEST for refs)  [$ROUND/gen/cands.tsv]
#   WAV_SCP      Kaldi dir or wav.scp (real / synth set)      []   (set CANDS_TSV=none to use it; `none` = unset)
#   MANIFEST     manifest TSV giving text_ref (required with CANDS_TSV; `none` = unset, only
#                allowed with WAV_SCP)                        [<pkg>/data/train_manifest.tsv]
#   OUT          output TSV (utt cand wav hyp ref n_ref edits mer)  [$ROUND/mer.tsv]
#   BATCH_SIZE   pipeline batch size                          [8]
#   LANGUAGE     force decoding language; empty or none = model default []
#   LIMIT        first N rows (0 = all)                       [0]
#   CAND_LABEL   cand value for WAV_SCP rows (synth for a 20_synthesize corpus)  [gt]
#   RESUME       1 = append to existing OUT                   [0]
#   JIWER_CHECK  1 = cross-check corpus MER with jiwer        [0]
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
conda activate "${CMI_DPO_ENV_ASR:?set CMI_DPO_ENV_ASR in config/paths.env}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$PKG"
export PYTHONPATH="${CMI_DPO_COSY_ROOT:?set CMI_DPO_COSY_ROOT in config/paths.env}:$CMI_DPO_COSY_ROOT/Matcha-TTS${PYTHONPATH:+:$PYTHONPATH}"

: "${ROUND:=$PKG/exp/round1}"
: "${WAV_SCP:=}"
if [[ "$WAV_SCP" == "none" ]]; then WAV_SCP=; fi
if [[ -n "$WAV_SCP" ]]; then : "${CANDS_TSV:=}"; else : "${CANDS_TSV:=$ROUND/gen/cands.tsv}"; fi
if [[ "$CANDS_TSV" == "none" ]]; then CANDS_TSV=; fi
: "${MANIFEST:=$PKG/data/train_manifest.tsv}"
: "${OUT:=$ROUND/mer.tsv}"
: "${BATCH_SIZE:=8}"
: "${LANGUAGE:=}"
: "${LIMIT:=0}"
: "${CAND_LABEL:=gt}"
: "${RESUME:=0}"
: "${JIWER_CHECK:=0}"
: "${MAX_FAIL_FRAC:=0.01}"
: "${STRICT:=0}"
: "${EXTRA:=}"
none_is_empty() { local v; for v in "$@"; do if [[ "${!v:-}" == "none" ]]; then printf -v "$v" ''; fi; done; }
none_is_empty LANGUAGE EXTRA MANIFEST

args=(--out "$OUT" --batch_size "$BATCH_SIZE" --limit "$LIMIT" --cand_label "$CAND_LABEL" --max_fail_frac "$MAX_FAIL_FRAC")
if [[ -n "$CANDS_TSV" && -n "$WAV_SCP" ]]; then echo "set only one of CANDS_TSV / WAV_SCP" >&2; exit 1; fi
if [[ -n "$CANDS_TSV" ]]; then
  if [[ -z "$MANIFEST" ]]; then echo "CANDS_TSV needs MANIFEST for the references (MANIFEST=none is only valid with WAV_SCP)" >&2; exit 1; fi
  args+=(--cands_tsv "$CANDS_TSV" --manifest "$MANIFEST")
elif [[ -n "$WAV_SCP" ]]; then
  args+=(--wav_scp "$WAV_SCP")
  if [[ -n "$MANIFEST" && -f "$MANIFEST" ]]; then args+=(--manifest "$MANIFEST"); fi
else
  echo "one of CANDS_TSV / WAV_SCP is required" >&2; exit 1
fi
if [[ -n "$LANGUAGE" ]]; then args+=(--language "$LANGUAGE"); fi
if [[ "$RESUME" == "1" ]]; then args+=(--resume); fi
if [[ "$JIWER_CHECK" == "1" ]]; then args+=(--jiwer_check); fi
if [[ "$STRICT" == "1" ]]; then args+=(--strict); fi

mkdir -p "$(dirname "$OUT")"
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-none} gpus=${CUDA_VISIBLE_DEVICES:-?}"
echo "python -u scripts/11_score_mer.py ${args[*]} $EXTRA"
# shellcheck disable=SC2086
python -u scripts/11_score_mer.py "${args[@]}" $EXTRA
echo 'MER_DONE'
