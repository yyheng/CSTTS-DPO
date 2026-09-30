#!/bin/bash
# ---------------------------------------------------------------------------
# run_dpo_round.sh : submit one multi-critic DPO round as a dependency chain (login-node safe:
# only sbatch/squeue/awk are called here; every python invocation runs inside the sbatch jobs).
#
#   [cmi gt]  (only if GT_CMI is missing or has no .done marker; 1 GPU; RESUME=1 when partial)
#   gen shard 0..NSHARDS-1     (NSHARDS x 1 GPU, concurrently)
#      └─ afterok ─► gen merge (NSHARDS > 1 only; CPU, --gres=gpu:0) -> gen/cands.tsv
#            └─ afterok ─► mer, utmos, cmi(cands)      (3 x 1 GPU)
#                        └─ afterok ─► pairs (CPU)
#                                        └─ afterok ─► dpo (DPO_GPUS GPUs, torchrun)
#
# Parameters (environment variables):
#   MANIFEST   manifest TSV from 00 (targets + prompts)      [<pkg>/data/train_manifest.tsv]
#   OUT        round dir: gen/ mer.tsv utmos.tsv cmi_cands.tsv pairs/ dpo/ jobs.txt  [<pkg>/exp/round1]
#   CKPT       LLM used for candidate generation AND as DPO init (SFT ckpt for round 1,
#              previous dpo_best.pth for later rounds); CKPT=none = the stock CosyVoice2 llm.pt for
#              both (gen.sbatch gets CKPT=none -> `--ckpt none`; dpo.sbatch gets INIT_CKPT=none ->
#              `--init_ckpt none`)   [$CMI_DPO_SFT_CKPT from config/paths.env; none when that is empty]
#   LAL_CKPT   WhisperWithLAL checkpoint (pickle or exported state_dict) for the CMI critic (cmi.sbatch LAL_CKPT)
#                                                            [$CMI_DPO_LAL_CKPT from config/paths.env]
#   CMI_BATCH  utterances per LAL encoder forward           [8]
#   MER_BATCH_SIZE  whisper pipeline batch size (mer.sbatch BATCH_SIZE)  [8]
#   N_CAND     candidates per utterance                      [4]
#   TEMP       sampling temperature                          [1.0]
#   GEN_TOP_P GEN_TOP_K  RAS overrides for gen; none = model default [none none]
#   SAMPLING   Qwen2LM.sampling_ids `sampling`               [25]
#   GEN_MAX_RETRIES  sampler retries per candidate           [2]
#   GEN_LIMIT  rows per gen shard (0 = all; smoke tests)     [0]
#   FORCE_RESUME  1 = gen shards resume despite a gen_config.json mismatch  [0]
#   LAM GAM NU critic weights (UTMOS, MER, dCMI); 0 disables the critic: it neither scores nor
#              gates (14_build_pairs.py) and its TSV is informational only   [1.0 1.0 1.0]
#   EXP        DPO experiment dir                            [$OUT/dpo]
#   BETA       DPO beta                                      [0.1]
#   NSHARDS    gen shards = concurrent GPUs for generation   [2]   (1..6)
#   DPO_GPUS   GPUs for the DPO job                          [2]
#   GT_CMI     cmi TSV of the ground-truth manifest rows     [<manifest without .tsv>.cmi_gt.tsv]
#   SEED                                                     [0]
#   MAX_MER MIN_UTMOS MAX_DCMI  pair thresholds on the preferred candidate  [0.20 2.5 0.20]
#   NORM       global | per_utt                              [global]
#   MER_CLIP   clip MER to [0, MER_CLIP] before normalisation (inf disables)  [1.0]
#   MAX_CAND_TOKENS  ranking-pool filter on speech tokens (750 = 30 s)   [750]
#   THRESHOLD_DISABLED  1 = also apply the thresholds of weight-0 critics [0]
#   KEEP_TRUNCATED  1 = keep candidates cut by max_len without EOS in the pool  [0]
#   MAX_FAIL_FRAC STRICT  critic failure policy (mer/utmos/cmi)   [0.01 0]
#   EPOCHS LR BATCH DPO_ACCUM  DPO hyper-parameters          [2 1e-6 4 1]
#   VAL_FRAC   held-out pair fraction for dpo_best.pth selection (0 = none)  [0.05]
#   MAX_PAIR_TOKENS  skip pairs whose pos/neg has more speech tokens (<= 0 off)  [750]
#   GRAD_CLIP BF16 SFT_WEIGHT LENGTH_NORM CACHE_REF LOG_EVERY DPO_SAVE_EVERY DPO_MAX_STEPS
#              remaining dpo.sbatch knobs                    [1.0 1 0 0 1 10 0 0]
#   DPO_RESUME checkpoint in EXP to resume the DPO job from; none = fresh  [none]
#   GEN_EXTRA MER_EXTRA UTMOS_EXTRA CMI_EXTRA PAIRS_EXTRA DPO_EXTRA
#              per-stage EXTRA flags (verbatim CLI, multi-word allowed: "--flag value --other");
#              none = [] [none ...]
#   DRY_RUN    1 = print the sbatch commands only            [0]
#   FORCE      1 = submit despite the GPU-budget warning     [0]
#
# Site configuration: PKG is derived from this file's location; config/paths.env is loaded through
# slurm/load_env.sh (CMI_DPO_* variables already set in the shell win over the file) for the CKPT /
# LAL_CKPT defaults, and every job is submitted through slurm/sb, which adds
# the site options (partition / exclude / account / extra / --output) and CMI_DPO_PKG. Each --export list
# below ALSO names CMI_DPO_PKG explicitly, because a caller --export replaces the one slurm/sb passes.
#
# Environment pinning: sbatch --export=ALL also forwards this shell's environment, so a stray
# LIMIT / EXTRA / RESUME / MAX_STEPS / SAVE_EVERY / ACCUM / TOP_P / TOP_K / CKPT / NSHARDS in
# the caller's shell would otherwise be picked up by an sbatch under the same name. Therefore EVERY
# variable each sbatch reads (its `: "${VAR:=` lines) is passed explicitly in that stage's --export
# list; values that must be empty are passed as the `none` sentinel (sbatch drops `VAR=`), which the
# sbatch files map back to "" (except INIT_CKPT=none, which means the stock model). Every per-stage export
# string is expanded QUOTED ("$gen_common" ...) so that a multi-word *_EXTRA stays inside the single
# --export argv word (Slurm keeps spaces inside one --export word). Driver knobs that
# share a name with a per-stage sbatch variable are prefixed (GEN_LIMIT GEN_TOP_P GEN_TOP_K GEN_EXTRA
# DPO_ACCUM DPO_MAX_STEPS DPO_SAVE_EVERY DPO_RESUME ...), so the bare names never reach any job.
#
# Example (paper setting, all three critics):
#   MANIFEST=$PWD/data/train_100h.tsv OUT=$PWD/exp/round1 NSHARDS=4 bash slurm/run_dpo_round.sh   (from the repo root)
# ---------------------------------------------------------------------------
set -euo pipefail
PKG=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SL=$PKG/slurm
SB=$SL/sb
MAX_GPUS=6
ENV_FILE=$PKG/config/paths.env
[[ -f "$ENV_FILE" ]] || { echo "$ENV_FILE not found; copy config/paths.env.example to it and fill in your paths" >&2; exit 1; }
# shellcheck disable=SC1091
source "$PKG/slurm/load_env.sh"   # CMI_DPO_* already in the environment win over the file
export CMI_DPO_PKG=$PKG

: "${MANIFEST:=$PKG/data/train_manifest.tsv}"
: "${OUT:=$PKG/exp/round1}"
: "${CKPT:=${CMI_DPO_SFT_CKPT:-none}}"        # empty CMI_DPO_SFT_CKPT = stock CosyVoice2 llm.pt (none)
: "${LAL_CKPT:=${CMI_DPO_LAL_CKPT:-}}"
: "${CMI_BATCH:=8}"
: "${MER_BATCH_SIZE:=8}"
: "${N_CAND:=4}"
: "${TEMP:=1.0}"
: "${GEN_TOP_P:=none}"
: "${GEN_TOP_K:=none}"
: "${SAMPLING:=25}"
: "${GEN_MAX_RETRIES:=2}"
: "${GEN_LIMIT:=0}"
: "${FORCE_RESUME:=0}"
: "${LAM:=1.0}"
: "${GAM:=1.0}"
: "${NU:=1.0}"
: "${EXP:=$OUT/dpo}"
: "${BETA:=0.1}"
: "${NSHARDS:=2}"
: "${DPO_GPUS:=2}"
: "${GT_CMI:=${MANIFEST%.tsv}.cmi_gt.tsv}"
: "${SEED:=0}"
: "${MAX_MER:=0.20}"
: "${MIN_UTMOS:=2.5}"
: "${MAX_DCMI:=0.20}"
: "${NORM:=global}"
: "${MER_CLIP:=1.0}"
: "${MAX_CAND_TOKENS:=750}"
: "${THRESHOLD_DISABLED:=0}"
: "${KEEP_TRUNCATED:=0}"
: "${MAX_FAIL_FRAC:=0.01}"
: "${STRICT:=0}"
: "${EPOCHS:=2}"
: "${LR:=1e-6}"
: "${BATCH:=4}"
: "${DPO_ACCUM:=1}"
: "${VAL_FRAC:=0.05}"
: "${MAX_PAIR_TOKENS:=750}"
: "${GRAD_CLIP:=1.0}"
: "${BF16:=1}"
: "${SFT_WEIGHT:=0}"
: "${LENGTH_NORM:=0}"
: "${CACHE_REF:=1}"
: "${LOG_EVERY:=10}"
: "${DPO_SAVE_EVERY:=0}"
: "${DPO_MAX_STEPS:=0}"
: "${DPO_RESUME:=none}"
: "${GEN_EXTRA:=none}"
: "${MER_EXTRA:=none}"
: "${UTMOS_EXTRA:=none}"
: "${CMI_EXTRA:=none}"
: "${PAIRS_EXTRA:=none}"
: "${DPO_EXTRA:=none}"
: "${DRY_RUN:=0}"

# ---- sanity ---------------------------------------------------------------
[[ -f "$MANIFEST" ]] || { echo "MANIFEST not found: $MANIFEST" >&2; exit 1; }
[[ "$CKPT" == "none" || -f "$CKPT" ]] || { echo "CKPT not found: $CKPT" >&2; exit 1; }
[[ -n "$LAL_CKPT" && -f "$LAL_CKPT" ]] || { echo "LAL_CKPT not found: '$LAL_CKPT' (set CMI_DPO_LAL_CKPT in config/paths.env or pass LAL_CKPT=...)" >&2; exit 1; }
if ! [[ "$NSHARDS" =~ ^[0-9]+$ ]] || (( NSHARDS < 1 || NSHARDS > MAX_GPUS )); then
  echo "NSHARDS must be in 1..$MAX_GPUS (got $NSHARDS)" >&2; exit 1
fi
if (( DPO_GPUS < 1 || DPO_GPUS > MAX_GPUS )); then echo "DPO_GPUS must be in 1..$MAX_GPUS" >&2; exit 1; fi
for v in PKG MANIFEST OUT CKPT LAL_CKPT EXP GT_CMI GEN_EXTRA MER_EXTRA UTMOS_EXTRA CMI_EXTRA PAIRS_EXTRA DPO_EXTRA DPO_RESUME; do
  [[ "${!v}" == *,* ]] && { echo "$v must not contain a comma (sbatch --export separator): ${!v}" >&2; exit 1; }
done
# *_EXTRA holds verbatim CLI flags (spaces allowed: the per-stage export strings are passed to sbatch as ONE
# quoted argv word, and the sbatch files word-split $EXTRA themselves); anything else is almost surely a typo
for v in GEN_EXTRA MER_EXTRA UTMOS_EXTRA CMI_EXTRA PAIRS_EXTRA DPO_EXTRA; do
  [[ "${!v}" == "none" || "${!v}" == -* ]] || echo "WARNING: $v='${!v}' does not start with '-' (expected verbatim CLI flags or none)" >&2
done
mkdir -p "$OUT" "$PKG/logs"

# GPUs already requested by this user's jobs: running + pending, INCLUDING dependency-held jobs of an
# earlier round (they will start while this round is still running, so they count; the sum over a
# whole earlier chain over-estimates its true peak, though). The chain itself never exceeds MAX_GPUS
# at any stage: max(NSHARDS + gt, 3 + gt, DPO_GPUS) <= 6.
gpus_of() { squeue -u "$USER" -h "$@" -o '%b' 2>/dev/null | grep -o 'gpu:[0-9]*' | cut -d: -f2 | paste -sd+ - || true; }
busy=$(gpus_of); busy=$(( ${busy:-0} ))
running=$(gpus_of -t R); running=$(( ${running:-0} ))
held=$(squeue -u "$USER" -h -t PD -o '%b %r' 2>/dev/null | grep -i 'dependency' | grep -o 'gpu:[0-9]*' | cut -d: -f2 | paste -sd+ - || true)
held=$(( ${held:-0} ))
need_gt=0
gt_resume=0
is_zero() { awk -v x="$1" 'BEGIN { exit !(x + 0 == 0) }'; }   # float-aware "weight == 0"
if is_zero "$NU"; then
  :                                   # dCMI disabled: 14_build_pairs.py skips the gt-CMI merge, no gt job needed
elif [[ -s "$GT_CMI" && -f "$GT_CMI.done" ]]; then
  :                                   # complete (cmi.sbatch writes OUT.done after a clean CMI_DONE)
elif [[ -s "$GT_CMI" ]]; then
  need_gt=1; gt_resume=1              # partial (killed / exit-2 job) or made before the .done marker existed
else
  need_gt=1
fi
gt_after_gen=0   # gt scoring runs beside the gen shards unless that would exceed MAX_GPUS
(( need_gt && NSHARDS + 1 > MAX_GPUS )) && gt_after_gen=1
peak=$(( NSHARDS + (need_gt && !gt_after_gen) ))   # generation stage
(( peak < 3 + (need_gt && gt_after_gen) )) && peak=$(( 3 + (need_gt && gt_after_gen) ))   # critic stage
(( peak < DPO_GPUS )) && peak=$DPO_GPUS
if (( busy + peak > MAX_GPUS )); then
  echo "WARNING: $busy GPU(s) already requested by your queued jobs ($running running, $held pending on a dependency," >&2
  echo "         $((busy - running - held)) pending for other reasons); this round peaks at $peak -> $((busy + peak)) > $MAX_GPUS." >&2
  echo "         Dependency-held jobs of an earlier round are counted because they will run alongside this one;" >&2
  echo "         if you know that earlier chain's own peak + $peak <= $MAX_GPUS, set FORCE=1. Otherwise lower" >&2
  echo "         NSHARDS/DPO_GPUS or wait for the other jobs." >&2
  [[ "${FORCE:-0}" == "1" ]] || exit 1
fi

submit() {  # submit <label> <file.sbatch> <sbatch args...> ; prints the job id (submission goes through slurm/sb)
  local label=$1 file=$2; shift 2
  if [[ "$DRY_RUN" == "1" ]]; then echo "$SB $file --parsable $*" >&2; echo "0"; return; fi
  local id
  id=$("$SB" "$file" --parsable "$@") || { echo "sbatch failed for $label" >&2; exit 1; }
  id=${id%%;*}
  echo "$label $id" >> "$OUT/jobs.txt"
  echo "$id"
}
join() { local IFS=:; echo "$*"; }

# ---- per-stage environment pins (every variable the sbatch reads; see header) ---------------
# gen.sbatch: ROUND MANIFEST OUT CKPT N_CAND TEMP TOP_P TOP_K SAMPLING SEED SHARD NSHARDS LIMIT SAVE_16K
#             TEXT_FRONTEND MAX_RETRIES FORCE_RESUME DRY_RUN MERGE EXTRA
gen_common="CMI_DPO_PKG=$PKG,ROUND=$OUT,MANIFEST=$MANIFEST,OUT=$OUT/gen,CKPT=$CKPT,N_CAND=$N_CAND,TEMP=$TEMP,TOP_P=$GEN_TOP_P,TOP_K=$GEN_TOP_K,SAMPLING=$SAMPLING,SEED=$SEED,NSHARDS=$NSHARDS,LIMIT=$GEN_LIMIT,SAVE_16K=0,TEXT_FRONTEND=0,MAX_RETRIES=$GEN_MAX_RETRIES,FORCE_RESUME=$FORCE_RESUME,DRY_RUN=0,EXTRA=$GEN_EXTRA"
# mer.sbatch: ROUND WAV_SCP CANDS_TSV MANIFEST OUT BATCH_SIZE LANGUAGE LIMIT CAND_LABEL RESUME JIWER_CHECK MAX_FAIL_FRAC STRICT EXTRA
mer_common="CMI_DPO_PKG=$PKG,ROUND=$OUT,WAV_SCP=none,CANDS_TSV=$OUT/gen/cands.tsv,MANIFEST=$MANIFEST,OUT=$OUT/mer.tsv,BATCH_SIZE=$MER_BATCH_SIZE,LANGUAGE=none,LIMIT=0,CAND_LABEL=gt,RESUME=0,JIWER_CHECK=0,MAX_FAIL_FRAC=$MAX_FAIL_FRAC,STRICT=$STRICT,EXTRA=$MER_EXTRA"
# utmos.sbatch: ROUND WAV_SCP CANDS_TSV OUT LIMIT CAND_LABEL RESUME MAX_FAIL_FRAC STRICT EXTRA
utmos_common="CMI_DPO_PKG=$PKG,ROUND=$OUT,WAV_SCP=none,CANDS_TSV=$OUT/gen/cands.tsv,OUT=$OUT/utmos.tsv,LIMIT=0,CAND_LABEL=gt,RESUME=0,MAX_FAIL_FRAC=$MAX_FAIL_FRAC,STRICT=$STRICT,EXTRA=$UTMOS_EXTRA"
# cmi.sbatch: ROUND CANDS_TSV MANIFEST WAV_SCP CAND_LABEL GT_ALL OUT BATCH LAL_CKPT DUMP_LABELS COUNT_ALL_CLASSES
#             TEXT_CMI LIMIT SHARD NSHARDS RESUME PRINT_SUMMARY MAX_FAIL_FRAC STRICT EXTRA   (the variable ones are per job)
cmi_common="CMI_DPO_PKG=$PKG,ROUND=$OUT,WAV_SCP=none,CAND_LABEL=gt,GT_ALL=0,BATCH=$CMI_BATCH,LAL_CKPT=$LAL_CKPT,DUMP_LABELS=0,COUNT_ALL_CLASSES=0,TEXT_CMI=0,LIMIT=0,SHARD=0,NSHARDS=1,PRINT_SUMMARY=1,MAX_FAIL_FRAC=$MAX_FAIL_FRAC,STRICT=$STRICT,EXTRA=$CMI_EXTRA"
# pairs.sbatch: ROUND CANDS MER UTMOS CMI CMI_GT TOKENS_DIR OUT LAM GAM NU NORM MER_CLIP MAX_MER MIN_UTMOS MAX_DCMI
#               THRESHOLD_DISABLED MAX_CAND_TOKENS KEEP_TRUNCATED EXTRA
pairs_common="CMI_DPO_PKG=$PKG,ROUND=$OUT,CANDS=$OUT/gen/cands.tsv,MER=$OUT/mer.tsv,UTMOS=$OUT/utmos.tsv,CMI=$OUT/cmi_cands.tsv,CMI_GT=$GT_CMI,TOKENS_DIR=$OUT/gen/tokens,OUT=$OUT/pairs,LAM=$LAM,GAM=$GAM,NU=$NU,NORM=$NORM,MER_CLIP=$MER_CLIP,MAX_MER=$MAX_MER,MIN_UTMOS=$MIN_UTMOS,MAX_DCMI=$MAX_DCMI,THRESHOLD_DISABLED=$THRESHOLD_DISABLED,MAX_CAND_TOKENS=$MAX_CAND_TOKENS,KEEP_TRUNCATED=$KEEP_TRUNCATED,EXTRA=$PAIRS_EXTRA"
# dpo.sbatch: ROUND PAIRS EXP INIT_CKPT RESUME GPUS BETA LR BATCH ACCUM EPOCHS GRAD_CLIP BF16 SFT_WEIGHT LENGTH_NORM
#             CACHE_REF VAL_FRAC MAX_PAIR_TOKENS LOG_EVERY SAVE_EVERY MAX_STEPS SEED EXTRA
init_ckpt=$CKPT     # CKPT=none -> INIT_CKPT=none -> `--init_ckpt none` = stock llm.pt for policy + reference
dpo_common="CMI_DPO_PKG=$PKG,ROUND=$OUT,PAIRS=$OUT/pairs/pairs.pt,EXP=$EXP,INIT_CKPT=$init_ckpt,RESUME=$DPO_RESUME,GPUS=$DPO_GPUS,BETA=$BETA,LR=$LR,BATCH=$BATCH,ACCUM=$DPO_ACCUM,EPOCHS=$EPOCHS,GRAD_CLIP=$GRAD_CLIP,BF16=$BF16,SFT_WEIGHT=$SFT_WEIGHT,LENGTH_NORM=$LENGTH_NORM,CACHE_REF=$CACHE_REF,VAL_FRAC=$VAL_FRAC,MAX_PAIR_TOKENS=$MAX_PAIR_TOKENS,LOG_EVERY=$LOG_EVERY,SAVE_EVERY=$DPO_SAVE_EVERY,MAX_STEPS=$DPO_MAX_STEPS,SEED=$SEED,EXTRA=$DPO_EXTRA"

echo "round: OUT=$OUT MANIFEST=$MANIFEST CKPT=$CKPT LAL_CKPT=$LAL_CKPT N_CAND=$N_CAND TEMP=$TEMP LAM/GAM/NU=$LAM/$GAM/$NU BETA=$BETA NSHARDS=$NSHARDS DPO_GPUS=$DPO_GPUS MER_CLIP=$MER_CLIP MAX_CAND_TOKENS=$MAX_CAND_TOKENS VAL_FRAC=$VAL_FRAC MAX_PAIR_TOKENS=$MAX_PAIR_TOKENS"
echo "# $(date) round submitted with: MANIFEST=$MANIFEST CKPT=$CKPT LAL_CKPT=$LAL_CKPT N_CAND=$N_CAND TEMP=$TEMP LAM=$LAM GAM=$GAM NU=$NU BETA=$BETA NSHARDS=$NSHARDS MER_CLIP=$MER_CLIP MAX_CAND_TOKENS=$MAX_CAND_TOKENS THRESHOLD_DISABLED=$THRESHOLD_DISABLED VAL_FRAC=$VAL_FRAC MAX_PAIR_TOKENS=$MAX_PAIR_TOKENS" >> "$OUT/jobs.txt"

# ---- ground-truth CMI (once per manifest) ---------------------------------
gt_job=""
if (( need_gt )); then
  if (( ! gt_after_gen )); then
    gt_job=$(submit cmi_gt "$SL/cmi.sbatch" --job-name=cd_cmi_gt \
      --export=ALL,CANDS_TSV=none,MANIFEST="$MANIFEST",OUT="$GT_CMI",RESUME="$gt_resume","$cmi_common")
    echo "cmi(gt)   job $gt_job -> $GT_CMI$( (( gt_resume )) && echo ' (RESUME=1: completing a partial file)')"
  fi
elif is_zero "$NU"; then
  echo "cmi(gt)   skipped (NU=0: dCMI disabled)"
else
  echo "cmi(gt)   reuse $GT_CMI"
fi

# ---- candidate generation shards ------------------------------------------
gen_jobs=()
for (( i = 0; i < NSHARDS; i++ )); do
  j=$(submit "gen$i" "$SL/gen.sbatch" --job-name="cd_gen$i" --export=ALL,SHARD="$i",MERGE=0,"$gen_common")
  gen_jobs+=("$j")
done
gen_dep="afterok:$(join "${gen_jobs[@]}")"
echo "gen       jobs ${gen_jobs[*]} (shards 0..$((NSHARDS - 1)))"

# with NSHARDS > 1 each shard writes gen/cands.tsv.<i>; the merge pass (CPU only) builds gen/cands.tsv
if (( NSHARDS > 1 )); then
  merge_job=$(submit gen_merge "$SL/gen.sbatch" --job-name=cd_gen_merge --dependency="$gen_dep" \
    --gres=gpu:0 --cpus-per-task=2 --time=00:30:00 \
    --export=ALL,SHARD=0,MERGE=1,"$gen_common")
  gen_dep="afterok:$merge_job"
  echo "gen merge job $merge_job -> $OUT/gen/cands.tsv"
else
  merge_job=""
fi

if (( need_gt && gt_after_gen )); then
  gt_job=$(submit cmi_gt "$SL/cmi.sbatch" --job-name=cd_cmi_gt --dependency="$gen_dep" \
    --export=ALL,CANDS_TSV=none,MANIFEST="$MANIFEST",OUT="$GT_CMI",RESUME="$gt_resume","$cmi_common")
  echo "cmi(gt)   job $gt_job (after gen) -> $GT_CMI$( (( gt_resume )) && echo ' (RESUME=1: completing a partial file)')"
fi

# ---- critics ---------------------------------------------------------------
# a critic whose weight is 0 is still scored (cheap, keeps the tables complete for a later ablation
# with other weights); 14_build_pairs.py ignores its values in R and in the gates
mer_job=$(submit mer "$SL/mer.sbatch" --job-name=cd_mer --dependency="$gen_dep" --export=ALL,"$mer_common")
utmos_job=$(submit utmos "$SL/utmos.sbatch" --job-name=cd_utmos --dependency="$gen_dep" --export=ALL,"$utmos_common")
cmi_job=$(submit cmi "$SL/cmi.sbatch" --job-name=cd_cmi --dependency="$gen_dep" \
  --export=ALL,CANDS_TSV="$OUT/gen/cands.tsv",MANIFEST=none,OUT="$OUT/cmi_cands.tsv",RESUME=0,"$cmi_common")
echo "critics   mer $mer_job  utmos $utmos_job  cmi $cmi_job"

# ---- pairs -----------------------------------------------------------------
pair_deps=("$mer_job" "$utmos_job" "$cmi_job")
[[ -n "$gt_job" ]] && pair_deps+=("$gt_job")
pairs_job=$(submit pairs "$SL/pairs.sbatch" --job-name=cd_pairs --dependency="afterok:$(join "${pair_deps[@]}")" \
  --export=ALL,"$pairs_common")
echo "pairs     job $pairs_job"

# ---- DPO -------------------------------------------------------------------
dpo_job=$(submit dpo "$SL/dpo.sbatch" --job-name=cd_dpo --dependency="afterok:$pairs_job" --gres="gpu:$DPO_GPUS" \
  --export=ALL,"$dpo_common")
echo "dpo       job $dpo_job -> $EXP/dpo_best.pth"

echo "job ids recorded in $OUT/jobs.txt ; logs in $PKG/logs/cd_*_<jobid>.out"
echo "ROUND_SUBMITTED gt=${gt_job:-reuse} gen=${gen_jobs[*]} merge=${merge_job:-none} mer=$mer_job utmos=$utmos_job cmi=$cmi_job pairs=$pairs_job dpo=$dpo_job"
