#!/bin/bash
#SBATCH --job-name=cd_prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
# ---------------------------------------------------------------------------
# 00_prep_manifest.py : Kaldi dir -> manifest TSV (CPU only, env $CMI_DPO_ENV_MAIN)
# Outputs next to OUT: <stem>.json (counts, hours available/selected, hours_cap_selects_all, prompt
# counts) and <stem>.dropped.txt (utt<TAB>reason for every dropped target; may be empty).
# With PROMPT_MODE=same_spk a target whose speaker has no other prompt utterance is DROPPED by default
# (ALLOW_SELF_PROMPT=1 uses the utterance itself instead, i.e. prompt == target).
# SEAME train filtered to [MIN_DUR, MAX_DUR] = [1, 30] s is ~94.3 h, so HOURS=100 selects all of it
# (the log and json say so: hours_cap_selects_all).
#
# Parameters (environment variables, pass with slurm/sb slurm/prep.sbatch --export=ALL,VAR=value,...):
#   DATA_DIR     Kaldi dir (wav.scp text utt2spk utt2dur)   [$CMI_DPO_DATA_ROOT/train from config/paths.env]
#   OUT          output manifest TSV                        [<pkg>/data/train_manifest.tsv]
#   TEXT_MODE    strip | raw : text_tts/prompt_text convention (docs/DESIGN.md)  [strip]
#                strip = <noise>/<unk> tags removed; raw = text verbatim incl. tags = the convention
#                the authors' stage-1 model SEAME_Final/epoch_020.pth was trained on
#   PROMPT_MODE  same_spk | self                            [same_spk]
#   PROMPT_POOL  all | subset : draw prompts from the whole Kaldi dir or only from the selected targets  [all]
#   ALLOW_SELF_PROMPT  1 = same_spk fallback to the target itself instead of dropping it  [0]
#   PROMPT_MIN   min prompt duration (s)                    [3]
#   PROMPT_MAX   max prompt duration (s)                    [10]
#   MIN_DUR      drop targets shorter than (s)              [1]
#   MAX_DUR      drop targets longer than (s)               [30]
#   HOURS        seeded subset cap in hours (> 0); empty or none = all    []
#   SEED         RNG seed                                   [0]
#   LIMIT        keep only the first N rows; empty or none = all  []
#   EXTRA        extra CLI flags appended verbatim; none = []    []
# Example: slurm/sb slurm/prep.sbatch --export=ALL,HOURS=100,OUT=$PWD/data/train_100h.tsv
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

: "${DATA_DIR:=$CMI_DPO_DATA_ROOT/train}"
: "${OUT:=$PKG/data/train_manifest.tsv}"
: "${TEXT_MODE:=strip}"
: "${PROMPT_MODE:=same_spk}"
: "${PROMPT_POOL:=all}"
: "${ALLOW_SELF_PROMPT:=0}"
: "${PROMPT_MIN:=3}"
: "${PROMPT_MAX:=10}"
: "${MIN_DUR:=1}"
: "${MAX_DUR:=30}"
: "${HOURS:=}"
: "${SEED:=0}"
: "${LIMIT:=}"
: "${EXTRA:=}"
none_is_empty() { local v; for v in "$@"; do if [[ "${!v:-}" == "none" ]]; then printf -v "$v" ''; fi; done; }
none_is_empty HOURS LIMIT EXTRA

args=(--data_dir "$DATA_DIR" --out "$OUT" --text_mode "$TEXT_MODE" --prompt_mode "$PROMPT_MODE"
      --prompt_pool "$PROMPT_POOL" --prompt_min "$PROMPT_MIN" --prompt_max "$PROMPT_MAX"
      --min_dur "$MIN_DUR" --max_dur "$MAX_DUR" --seed "$SEED")
if [[ "$ALLOW_SELF_PROMPT" == "1" ]]; then args+=(--allow_self_prompt); fi
if [[ -n "$HOURS" ]]; then args+=(--hours "$HOURS"); fi
if [[ -n "$LIMIT" ]]; then args+=(--limit "$LIMIT"); fi

mkdir -p "$(dirname "$OUT")"
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "python -u scripts/00_prep_manifest.py ${args[*]} $EXTRA"
# shellcheck disable=SC2086
python -u scripts/00_prep_manifest.py "${args[@]}" $EXTRA
echo 'PREP_DONE'
