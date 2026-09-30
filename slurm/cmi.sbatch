#!/bin/bash
#SBATCH --job-name=cd_cmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
# ---------------------------------------------------------------------------
# 13_score_cmi.py : CMIspeech critic (Whisper-LAL frame labels), 1 GPU, env $CMI_DPO_ENV_LAL
# At least one of CANDS_TSV / MANIFEST / WAV_SCP must be non-empty:
#   CANDS_TSV only  -> candidate rows            (round critic:  OUT = $ROUND/cmi_cands.tsv)
#   MANIFEST only   -> ground-truth rows (cand=gt) (once per manifest: OUT = <manifest>.cmi_gt.tsv)
#   both            -> gt + candidates in one file (what 14_build_pairs.py / eval_tts.py consume);
#                      gt rows are then restricted to utts that have a candidate (GT_ALL=1 keeps all)
#   WAV_SCP         -> Kaldi dir / wav.scp (devman, or a 20_synthesize.py output) with cand=CAND_LABEL;
#                      WAV_SCP + MANIFEST + CAND_LABEL=synth + GT_ALL=1 -> gt + synthetic rows in one file
#                      (GT_ALL=1 is needed because the synth utts carry the syn_ prefix and would otherwise
#                      match no gt row; eval_tts.py strips the prefix for the dCMI join)
# 14_build_pairs.py needs gt AND candidate rows in ONE file; pairs.sbatch merges CMI + CMI_GT.
# Failed rows (unreadable wav, ...) are listed in OUT.failed.txt and skipped: exit status 0 (with a
# WARNING) while failed / rows in scope <= MAX_FAIL_FRAC (default 1 %), exit 2 (job FAILED, the afterok
# chain stops, no .done marker) above it or with STRICT=1. Re-run with RESUME=1 to retry exactly the
# failed rows. A run that exits 0 writes the marker OUT.done next to OUT (absent after a killed / exit-2 job).
#
# Parameters:
#   ROUND        round dir used for the defaults              [<pkg>/exp/round1]
#   CANDS_TSV    one or more cands.tsv (space separated)      [$ROUND/gen/cands.tsv]  (CANDS_TSV=none to skip)
#   MANIFEST     manifest TSV whose wav column is scored as cand=gt   []   (MANIFEST=none = unset)
#   WAV_SCP      Kaldi dir (or wav.scp) scored with cand=CAND_LABEL    []   (WAV_SCP=none = unset)
#   (use the `none` sentinel, never `VAR=`: sbatch drops an empty assignment and srun even keeps the
#    caller's exported value)
#   CAND_LABEL   cand value for WAV_SCP rows                 [gt]   (must differ from gt when MANIFEST is also set)
#   GT_ALL       1 = keep gt rows of utts without candidates [0]
#   OUT          output TSV                                   [$ROUND/cmi_cands.tsv]
#   BATCH        utterances per encoder forward               [8]
#   LAL_CKPT     WhisperWithLAL checkpoint (pickle or exported state_dict)  [$CMI_DPO_LAL_CKPT from config/paths.env]
#                (deliberately NOT named CKPT: that name is the TTS LLM checkpoint of gen/synth/run_dpo_round.sh
#                 and would leak in through --export=ALL)
#   DUMP_LABELS  1 = add labels_rle column                    [0]
#   COUNT_ALL_CLASSES 1 = count blank/other in T(u)           [0]   (default L = {zh, en})
#   TEXT_CMI     1 = add the text_cmi column (token-level CMI of the manifest text_tts; needs MANIFEST) [0]
#   LIMIT        first N utterances (0 = all)                 [0]
#   SHARD / NSHARDS  over utterances                          [0 / 1]
#   RESUME       1 = append to existing OUT                   [0]
#   PRINT_SUMMARY 1 = print per-cand mean CMI / dCMI          [1]
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
conda activate "${CMI_DPO_ENV_LAL:?set CMI_DPO_ENV_LAL in config/paths.env}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$PKG"
export PYTHONPATH="${CMI_DPO_COSY_ROOT:?set CMI_DPO_COSY_ROOT in config/paths.env}:$CMI_DPO_COSY_ROOT/Matcha-TTS${PYTHONPATH:+:$PYTHONPATH}"

: "${ROUND:=$PKG/exp/round1}"
: "${CANDS_TSV=$ROUND/gen/cands.tsv}"
if [[ "$CANDS_TSV" == "none" ]]; then CANDS_TSV=; fi
: "${MANIFEST:=}"
if [[ "$MANIFEST" == "none" ]]; then MANIFEST=; fi
: "${WAV_SCP:=}"
if [[ "$WAV_SCP" == "none" ]]; then WAV_SCP=; fi
: "${CAND_LABEL:=gt}"
: "${GT_ALL:=0}"
: "${OUT:=$ROUND/cmi_cands.tsv}"
: "${BATCH:=8}"
: "${LAL_CKPT:=${CMI_DPO_LAL_CKPT:-}}"   # config/paths.env CMI_DPO_LAL_CKPT
[[ -n "$LAL_CKPT" ]] || { echo "LAL_CKPT is empty: set CMI_DPO_LAL_CKPT in config/paths.env or pass LAL_CKPT=..." >&2; exit 1; }
: "${DUMP_LABELS:=0}"
: "${COUNT_ALL_CLASSES:=0}"
: "${TEXT_CMI:=0}"
: "${LIMIT:=0}"
: "${SHARD:=0}"
: "${NSHARDS:=1}"
: "${RESUME:=0}"
: "${PRINT_SUMMARY:=1}"
: "${MAX_FAIL_FRAC:=0.01}"
: "${STRICT:=0}"
: "${EXTRA:=}"
if [[ "$EXTRA" == "none" ]]; then EXTRA=; fi

if [[ -z "$CANDS_TSV" && -z "$MANIFEST" && -z "$WAV_SCP" ]]; then echo "at least one of CANDS_TSV / MANIFEST / WAV_SCP is required" >&2; exit 1; fi
args=(--out "$OUT" --ckpt "$LAL_CKPT" --batch "$BATCH" --limit "$LIMIT" --shard "$SHARD" --nshards "$NSHARDS"
      --device cuda:0 --cand_label "$CAND_LABEL" --max_fail_frac "$MAX_FAIL_FRAC")
# shellcheck disable=SC2206
if [[ -n "$CANDS_TSV" ]]; then args+=(--cands_tsv $CANDS_TSV); fi
if [[ -n "$MANIFEST" ]]; then args+=(--manifest "$MANIFEST"); fi
if [[ -n "$WAV_SCP" ]]; then args+=(--wav_scp "$WAV_SCP"); fi
if [[ "$GT_ALL" == "1" ]]; then args+=(--gt_all); fi
if [[ "$DUMP_LABELS" == "1" ]]; then args+=(--dump_labels); fi
if [[ "$COUNT_ALL_CLASSES" == "1" ]]; then args+=(--count_all_classes); fi
if [[ "$TEXT_CMI" == "1" ]]; then args+=(--text_cmi); fi
if [[ "$RESUME" == "1" ]]; then args+=(--resume); fi
if [[ "$PRINT_SUMMARY" == "1" ]]; then args+=(--print_summary); fi
if [[ "$STRICT" == "1" ]]; then args+=(--strict); fi

mkdir -p "$(dirname "$OUT")"
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-none} gpus=${CUDA_VISIBLE_DEVICES:-?}"
echo "python -u scripts/13_score_cmi.py ${args[*]} $EXTRA"
# shellcheck disable=SC2086
python -u scripts/13_score_cmi.py "${args[@]}" $EXTRA
touch "$OUT.done"   # completion marker (run_dpo_round.sh reuses GT_CMI only when it exists)
echo 'CMI_DONE'
