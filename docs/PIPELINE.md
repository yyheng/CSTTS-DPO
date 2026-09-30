# Pipeline: data, training, synthesis and evaluation

Recipes for the full pipeline (paper sections 3-4). Everything is driven through the SLURM layer
described in [`CLUSTER.md`](CLUSTER.md): `slurm/sb <file.sbatch> [sbatch options]` submits one job
with the site settings of `config/paths.env`, and each sbatch file documents its variables and
defaults in its header (`head -60 slurm/<x>.sbatch`). The same python commands can be run directly
in the `cstts` env on a single machine (`python scripts/<x>.py --help`), which is what the sbatch
files do internally.

Conventions used below:

```bash
cd /path/to/CSTTS-DPO           # repository root
PKG=$(pwd)
source slurm/load_env.sh                  # only needed for $CMI_DPO_* in the shell snippets (keeps variables you exported)
```

Package layout: `data/` (manifests + SFT token caches), `exp/` (`sft_seame/`,
`round<k>/{gen,mer.tsv,utmos.tsv,cmi_cands.tsv,pairs,dpo}`, `synth_*/`), `logs/` (SLURM logs
`<job-name>_<jobid>.out`). All three are git-ignored.

Parameters of a job are environment variables passed with `--export=ALL,VAR=value,...`.
`EXTRA="--flag value"` appends arbitrary CLI flags (multi-word values are fine: Slurm keeps spaces
inside one `--export` word, the sbatch files word-split `$EXTRA`, and `run_dpo_round.sh` passes its
per-stage export strings quoted). `sbatch` drops an empty assignment (`VAR=`), so every sbatch accepts
the sentinel **`VAR=none`** for "empty" (`CANDS_TSV=none`, `TOP_P=none`, `EXTRA=none`, `RESUME=none`,
...; each header lists the variables it maps); the exceptions are `INIT_CKPT=none` of `dpo.sbatch`,
which means "start from the stock CosyVoice2 `llm.pt`", and `MANIFEST=none` of `mer.sbatch`, which is
only valid with `WAV_SCP` (a `CANDS_TSV` run needs the manifest for its references and errors out
otherwise).

## 1. Data conventions

Input: Kaldi dirs `$CMI_DPO_DATA_ROOT/{train,valid,devman,devsge}` (`wav.scp text utt2spk utt2dur`,
16 kHz mono audio). `text` is lowercase, Chinese one character per space-separated token, and may hold
`<noise>` / `<unk>` tags. Utterance ids are `<spk>-...`.

Three text views used everywhere:

| column | definition | used by |
|---|---|---|
| `text_raw` | transcript as in `text` | synth `text` file |
| `text_tts` | `TEXT_MODE=strip` (default): `text_raw` with `<...>` tags removed, whitespace collapsed; `TEXT_MODE=raw`: `text_raw` verbatim incl. `<noise>`/`<unk>` (the convention the authors' stage-1 checkpoint was trained on). SEAME spacing is kept either way; `prompt_text` follows the same mode and 10/20 take both columns verbatim | TTS LLM input (SFT, gen, synth), text CMI |
| `text_ref` | `seame_normalize(text_raw)` (lowercase, tags to space, opencc t2s, apostrophes dropped, CJK chars isolated, punctuation to space) | MER reference |

**Manifest TSV** (output of 00, tab-separated with header):
`utt spk dur wav text_raw text_tts text_ref prompt_utt prompt_wav prompt_text`, plus the side files
`<stem>.json` (counts, hours available / selected, `hours_cap_selects_all`, prompt counts) and
`<stem>.dropped.txt` (`utt<TAB>reason` per dropped target, may be empty).
Prompt = a *different utterance of the same speaker* with duration in [3, 10] s drawn with a seeded
RNG (fallback: the other utt of the speaker closest to that window, from a pool restricted to
[1, 30] s; `PROMPT_POOL=all` (default) draws from the whole Kaldi dir, `subset` only from the selected
targets). A target whose speaker has **no other pool utterance is dropped** (reason
`no_other_prompt_utt`; `ALLOW_SELF_PROMPT=1` uses the utterance itself instead, which leaks the target
into the prompt). Targets are filtered to [1, 30] s and non-empty `text_tts`.
`HOURS=H` draws a seeded random subset up to H hours. SEAME train filtered to [1, 30] s is slightly
below 100 h, so `HOURS=100` selects all of it (`hours_cap_selects_all: true` in the json): the paper's
"100 h of real SEAME" is the whole filtered train here, and its "100 h real + 100 h synthetic" is
approximated by the full filtered train plus one synthesis pass over the same transcripts (section 6).

## 2. Manifests (CPU)

```bash
for s in train valid devman devsge; do
  slurm/sb slurm/prep.sbatch --export=ALL,DATA_DIR=$CMI_DPO_DATA_ROOT/$s,OUT=$PKG/data/${s}_manifest.tsv
done
# -> data/<set>_manifest.{tsv,json,dropped.txt}; read the json: hours_selected, n_dropped_no_prompt, hours_cap_selects_all
```

`TEXT_MODE=raw` (section 1) keeps the `<noise>`/`<unk>` tags in `text_tts` / `prompt_text`, matching how
the authors' stage-1 model was trained; the default `strip` is this package's convention.
`10_gen_candidates.py` / `20_synthesize.py` take both columns from the manifest verbatim (no re-stripping)
and record the inferred mode in their config fingerprint. Other knobs: `PROMPT_POOL`, `ALLOW_SELF_PROMPT`,
`PROMPT_MIN/MAX`, `MIN_DUR/MAX_DUR`, `HOURS` (must be > 0), `SEED`, `LIMIT`.

## 3. Stage 1: SFT of the CosyVoice2 LLM (paper section 4)

`CMI_DPO_SFT_CKPT` (if set) is the default `CKPT` of stage 2, so this stage is only needed to
(re)train it. Token cache: 4 train shards + 1 valid = **5 GPUs at once**, then SFT on 4 GPUs **after**
all five caches finished (`--dependency=afterok`), so the recipe never holds more than 5 GPUs:

```bash
ids=""
for i in 0 1 2 3; do
  ids+=":$(slurm/sb slurm/sft_cache.sbatch --parsable --export=ALL,MANIFEST=$PKG/data/train_manifest.tsv,SHARD=$i,NSHARDS=4)"
done
ids+=":$(slurm/sb slurm/sft_cache.sbatch --parsable --export=ALL,MANIFEST=$PKG/data/valid_manifest.tsv)"
# -> data/sft_cache/train_manifest.shard{0..3}.pt (+ .json counts), data/sft_cache/valid_manifest.shard0.pt
slurm/sb slurm/sft.sbatch --dependency=afterok${ids}                                   # 4 GPUs, EXP=$PKG/exp/sft_seame
# paper-literal batch (4 sequences per optimizer step) on the same 4 GPUs:
slurm/sb slurm/sft.sbatch --dependency=afterok${ids} --export=ALL,GLOBAL_BATCH=4,BATCH=1,EXP=$PKG/exp/sft_seame_gb4
# other GPU count: override BOTH the allocation and GPUS
slurm/sb slurm/sft.sbatch --dependency=afterok${ids} --gres=gpu:2 --export=ALL,GPUS=2,EXP=$PKG/exp/sft_seame_2gpu
# then set CMI_DPO_SFT_CKPT=$PKG/exp/sft_seame/best.pth (or pass CKPT=...) in stage 2
```

Training = torchrun DDP over the LLM only; AdamW peak lr 2e-4 with linear warm-up then **constant** lr
(`LR_SCHEDULE=constant_with_warmup`, the paper only states a warm-up; `LR_SCHEDULE=linear` decays to 0),
up to 50k optimizer steps, validation every `EVAL_EVERY` optimizer steps, early stopping on val loss
(`PATIENCE`), checkpoints `<EXP>/best.pth` and `<EXP>/epoch_XXX.pth` (no `module.` prefix) plus
`<EXP>/train_state.pt` (~4 GB: optimizer, scheduler, counters of the last checkpoint) for
`RESUME=<EXP>/epoch_XXX.pth`.
**Batch:** `BATCH` is the micro-batch *per GPU*, so the effective batch is `BATCH x GPUS x ACCUM` =
4 x 4 = 16 with the defaults. The paper's "batch size of 4" is `GLOBAL_BATCH=4`: the script sets
`accum = GLOBAL_BATCH / (BATCH x GPUS)`, which must be an integer (hence `BATCH=1` on 4 GPUs, or `BATCH=4`
on 1 GPU), and overrides `ACCUM`. `MAX_STEPS`, `EVAL_EVERY`, `WARMUP_STEPS` count optimizer steps.

## 4. Stage 2: one multi-critic DPO round (paper section 3.2)

`slurm/run_dpo_round.sh` submits the whole round as a dependency chain and prints the job ids
(also recorded in `$OUT/jobs.txt`):

```
[cmi gt: 13 on the manifest wavs, only if NU != 0 and GT_CMI or its .done marker is missing]   1 GPU
gen shard 0..NSHARDS-1  (10, N_CAND candidates/utt at TEMP)      NSHARDS x 1 GPU
   └─afterok─► gen merge (10 --merge, NSHARDS > 1 only: cands.tsv.<i> -> cands.tsv)   CPU (--gres=gpu:0)
         └─afterok─► mer (11) | utmos (12) | cmi cands (13)           3 x 1 GPU
                  └─afterok─► pairs (14, CPU: merges gt+cand CMI, R = λ·UTMOS − γ·MER − ν·ΔCMI)
                                 └─afterok─► dpo (15, torchrun, DPO_GPUS GPUs)
```

```bash
MANIFEST=$PKG/data/train_manifest.tsv OUT=$PKG/exp/round1 \
CKPT=$CMI_DPO_SFT_CKPT \
N_CAND=4 TEMP=1.0 LAM=1.0 GAM=1.0 NU=1.0 BETA=0.1 NSHARDS=5 DPO_GPUS=2 \
bash slurm/run_dpo_round.sh
# -> exp/round1/gen/{wav,tokens,gen_config.json,cands.tsv.0..4 -> cands.tsv (merge job)}  mer.tsv  utmos.tsv  cmi_cands.tsv
#    data/train_manifest.cmi_gt.tsv + .done marker (reused by every later round on the same manifest;
#    a partial file without the marker is completed with RESUME=1 instead of being trusted)
#    exp/round1/pairs/{cmi_merged.tsv,pairs.pt,pairs_summary.json}
#    exp/round1/dpo/{dpo_best.pth,dpo_epoch_XXX.pth,metrics.jsonl,val_pairs.txt,train_state.pt,ref_cache.pt}
```

Driver variables (defaults in brackets; full list in the script header):
`EXP` (DPO dir, `$OUT/dpo`), `LAL_CKPT` / `CMI_BATCH` / `MER_BATCH_SIZE` (critic checkpoint and batch sizes),
`SEED`, `GEN_TOP_P` / `GEN_TOP_K` / `SAMPLING` / `GEN_MAX_RETRIES` / `GEN_LIMIT` / `FORCE_RESUME` (generation),
`MAX_MER MIN_UTMOS MAX_DCMI` (0.20 / 2.5 / 0.20), `NORM` (global|per_utt), **`MER_CLIP`** (1.0; `inf` disables),
**`MAX_CAND_TOKENS`** (750), **`THRESHOLD_DISABLED`** (0), `KEEP_TRUNCATED` (0), `MAX_FAIL_FRAC` / `STRICT`
(critic failure policy, 0.01 / 0), `EPOCHS LR BATCH DPO_ACCUM` (2 / 1e-6 / 4 / 1), **`VAL_FRAC`** (0.05),
**`MAX_PAIR_TOKENS`** (750), `GRAD_CLIP BF16 SFT_WEIGHT LENGTH_NORM CACHE_REF LOG_EVERY DPO_SAVE_EVERY DPO_MAX_STEPS`,
`DPO_RESUME`, per-stage `GEN_EXTRA MER_EXTRA UTMOS_EXTRA CMI_EXTRA PAIRS_EXTRA DPO_EXTRA` (verbatim CLI flags,
`"--flag value --other"` allowed; a value not starting with `-` only warns), `DRY_RUN=1`
(print the submit lines only), `FORCE=1` (submit despite the GPU-budget warning; the warning counts pending
dependency-held jobs of an earlier round too, so it is expected when chaining rounds whose own peaks fit
in the budget together).
**Environment pinning:** every variable an sbatch reads is passed explicitly in that stage's `--export`
list (empty values as `none`), so a stray `LIMIT`, `EXTRA`, `RESUME`, `MAX_STEPS`, `SAVE_EVERY`, `ACCUM`,
`TOP_P`, `TOP_K`, `CKPT` or `NSHARDS` in your shell cannot reach any job; driver knobs sharing a name with
a per-stage variable are prefixed (`GEN_*`, `DPO_*`). `CKPT=none` uses the stock CosyVoice2 `llm.pt` for
generation (`--ckpt none`) **and** as DPO policy/reference init (`--init_ckpt none`). `NU=0` skips the
gt-CMI job and the gt merge altogether.
A second round starts from the first round's model: `CKPT=$PKG/exp/round1/dpo/dpo_best.pth OUT=$PKG/exp/round2 ...`.

Re-running a stage by hand (each sbatch defaults to the `ROUND` layout above):

```bash
slurm/sb slurm/gen.sbatch --export=ALL,ROUND=$PKG/exp/round1,MANIFEST=$PKG/data/train_manifest.tsv,SHARD=0,NSHARDS=5   # resumable (-> gen/cands.tsv.0); FORCE_RESUME=1 if gen_config.json differs
slurm/sb slurm/gen.sbatch --export=ALL,ROUND=$PKG/exp/round1,MANIFEST=$PKG/data/train_manifest.tsv,SHARD=0,NSHARDS=5,DRY_RUN=1   # plan only (done/pending rows), no model load
slurm/sb slurm/gen.sbatch --gres=gpu:0 --cpus-per-task=2 --time=00:30:00 --export=ALL,ROUND=$PKG/exp/round1,MERGE=1   # after all shards: -> gen/cands.tsv (CPU)
slurm/sb slurm/mer.sbatch --export=ALL,ROUND=$PKG/exp/round1,MANIFEST=$PKG/data/train_manifest.tsv,RESUME=1
slurm/sb slurm/utmos.sbatch --export=ALL,ROUND=$PKG/exp/round1,RESUME=1
slurm/sb slurm/cmi.sbatch --export=ALL,ROUND=$PKG/exp/round1,RESUME=1                                   # candidates
slurm/sb slurm/cmi.sbatch --export=ALL,CANDS_TSV=none,MANIFEST=$PKG/data/train_manifest.tsv,OUT=$PKG/data/train_manifest.cmi_gt.tsv   # gt only
slurm/sb slurm/pairs.sbatch --export=ALL,ROUND=$PKG/exp/round1,CMI_GT=$PKG/data/train_manifest.cmi_gt.tsv
slurm/sb slurm/dpo.sbatch --export=ALL,ROUND=$PKG/exp/round1                                            # 2 GPUs
slurm/sb slurm/dpo.sbatch --export=ALL,ROUND=$PKG/exp/round1,RESUME=$PKG/exp/round1/dpo/dpo_epoch_001.pth   # resume a pre-empted DPO run
```

Every critic tolerates unreadable rows: they go to `<OUT>.failed.txt`, the job exits 0 with a WARNING while
`failed / rows <= MAX_FAIL_FRAC` (1 %), and exits 2 (chain stops, no `.done` marker for cmi) above it or with
`STRICT=1`; `RESUME=1` retries exactly the failed rows.

## 5. The paper's Table-1 rows by critic weights (ablation)

Pair building is cheap (CPU) and deterministic, so the critic ablation reuses one candidate pool
and one set of critic tables; only `14` and `15` are re-run. **A critic with weight 0 is disabled
entirely**: it contributes nothing to `R`, its threshold is *not* applied (unless
`THRESHOLD_DISABLED=1`) and its TSV is optional (passed only for information when present); with
`NU=0` no gt-CMI merge happens. So the rows differ only in the weights:

| Table-1 row | LAM (UTMOS) | GAM (MER) | NU (ΔCMI) |
|---|---|---|---|
| SFT only (no DPO) | - | - | - (evaluate `CKPT` directly, section 7) |
| DPO, MER only | 0 | 1 | 0 |
| DPO, MER + UTMOS | 1 | 1 | 0 |
| DPO, MER + UTMOS + ΔCMI (proposed) | 1 | 1 | 1 |

The three DPO jobs are chained one after another (`afterany`, so a failed row does not block the next),
which keeps the peak at **2 GPUs** (three concurrent DPO jobs would be 6):

```bash
R=$PKG/exp/round1; GT=$PKG/data/train_manifest.cmi_gt.tsv; prev=""
for w in "0 1 0 mer" "1 1 0 mer_utmos" "1 1 1 all3"; do set -- $w
  p=$(slurm/sb slurm/pairs.sbatch --parsable --job-name=cd_pairs_$4 --export=ALL,ROUND=$R,CMI_GT=$GT,OUT=$R/pairs_$4,LAM=$1,GAM=$2,NU=$3)
  dep="afterok:$p"; [[ -n "$prev" ]] && dep+=",afterany:$prev"
  prev=$(slurm/sb slurm/dpo.sbatch --parsable --job-name=cd_dpo_$4 --dependency=$dep --export=ALL,ROUND=$R,PAIRS=$R/pairs_$4/pairs.pt,EXP=$R/dpo_$4)
done   # peak 2 GPUs; drop the afterany link for 3 x 2 = 6 GPUs at once (only if nothing else of yours is queued)
```

Pass `MAX_DCMI=inf` etc. to relax a threshold of an *active* critic; `THRESHOLD_DISABLED=1` restores the
behaviour where a disabled critic still gates the preferred candidate.

## 6. Synthesis of the augmentation corpus (e.g. 100 h) with shards

`HOURS` is the **corpus-level** cap: `20_synthesize.py` stops each shard at `HOURS/NSHARDS` itself, so pass
the corpus target (e.g. 100) to every shard. The manifest rows are processed in a **seeded shuffled order**
(`SEED`; `NO_SHUFFLE=1` keeps manifest order), so shards and the capped subset are speaker-balanced.
Utterances whose generation hit `max_len` without EOS or whose wav exceeds `MAX_DUR` (30 s) are re-drawn
(`REJECT_RETRIES=1`) and otherwise listed in `failed.txt` as `rejected: ...` (`KEEP_TRUNCATED=1` /
`MAX_DUR=inf` disable the filters). On a resume those `rejected:` utts count as done (the draws are
deterministic, re-drawing them would only repeat the rejection; the DONE line reports them as
`rejected_skipped=`); `EXTRA=--retry_rejected` re-draws them (sensible only with a changed
`REJECT_RETRIES` / `KEEP_TRUNCATED` / `MAX_DUR`, i.e. together with `FORCE_RESUME=1`). Shards write
`wav.scp.<i>` etc. and a final `MERGE=1` run (a seconds-long CPU file merge; submit it with `--gres=gpu:0`
so no GPU is held) produces the Kaldi files and merges `failed.txt.<i>` into `failed.txt` (one line per
utt, utts that made it into the corpus removed). Peak = `NSHARDS` GPUs (6 below).

```bash
CK=$PKG/exp/round1/dpo/dpo_best.pth; O=$PKG/exp/synth_round1_100h; ids=""
for i in 0 1 2 3 4 5; do
  ids+=":$(slurm/sb slurm/synth.sbatch --parsable --job-name=cd_synth$i --export=ALL,MANIFEST=$PKG/data/train_manifest.tsv,CKPT=$CK,OUT=$O,HOURS=100,SHARD=$i,NSHARDS=6)"   # HOURS = corpus total
done
slurm/sb slurm/synth.sbatch --gres=gpu:0 --cpus-per-task=2 --time=00:30:00 --dependency=afterok${ids} --export=ALL,OUT=$O,MERGE=1   # CPU only
# -> $O/wav/syn_<utt>.wav (16 kHz), $O/{wav.scp,text,utt2spk,utt2dur}, $O/failed.txt.<i> -> failed.txt (merge), $O/synth_config.json
# each shard's log ends with: DONE shard=i/6 rows= done= skipped= failed= rejected= rejected_skipped= redraws= hours_in_shard= capped= out=
# the merge's with:          DONE merge out= utts= dropped_incomplete= failed_utts=
```

Output utts are prefixed `syn_` (`UTT_PREFIX`) so the corpus can be concatenated with real train
without id clashes. `HOURS=100` on the filtered train manifest synthesises every transcript once (= the
paper's "100 h synthetic"). Use `CKPT=<sft ckpt>` for the SFT-only baseline corpus and `CKPT=none` for the
stock model; `TEMP` and `SEED` control diversity; the result is resumable (re-submit the same shards; a
changed `CKPT`/`TEMP`/`SEED`/... or a manifest re-generated with another prompt assignment (content hash
`manifest_sha256`) is refused by the `synth_config.json` fingerprint unless `FORCE_RESUME=1`; `DRY_RUN=1`
prints done / rejected / pending / beyond-cap counts without loading the model).

## 7. Evaluating a TTS system (paper Table 1: UTMOS, MER, ΔCMI on devman / devsge)

Generate one candidate per test utterance with the model under test (1 GPU), score it with the three
critics (3 GPUs after it), score the real recordings once (gt CMI, and optionally gt MER/UTMOS for
the ground-truth row), then merge:

```bash
M=$PKG/data/devman_manifest.tsv; S=$PKG/exp/eval_devman_dpo1; CK=$PKG/exp/round1/dpo/dpo_best.pth
g=$(slurm/sb slurm/gen.sbatch --parsable --job-name=cd_gen_eval --export=ALL,MANIFEST=$M,OUT=$S/gen,CKPT=$CK,N_CAND=1,SEED=0)
m=$(slurm/sb slurm/mer.sbatch --parsable --dependency=afterok:$g --export=ALL,ROUND=$S,MANIFEST=$M)
u=$(slurm/sb slurm/utmos.sbatch --parsable --dependency=afterok:$g --export=ALL,ROUND=$S)
c=$(slurm/sb slurm/cmi.sbatch --parsable --dependency=afterok:$g --export=ALL,ROUND=$S,MANIFEST=$M,OUT=$S/cmi.tsv)   # gt + cand rows in one file
slurm/sb slurm/eval.sbatch --dependency=afterok:$m:$u:$c --export=ALL,ROUND=$S,SET_DIR=$S/gen,CMI_TSV=$S/cmi.tsv,CAND=0,SET_NAME=devman_dpo1
# -> $S/gen/eval.json and a one-line table: set UTMOS MER% CMI% CMIgt% dCMI% rows
```

Ground-truth row (real devman): give it its own directory. Without `ROUND`/`OUT` the critics default to
`exp/round1/{mer.tsv,utmos.tsv,cmi_cands.tsv}` and would **truncate round 1's candidate tables** (every
critic rewrites `OUT` unless `RESUME=1`). Skips use the `none` sentinel (`CANDS_TSV=none`, `MANIFEST=none`),
never `VAR=` (sbatch drops an empty assignment):

```bash
D=$CMI_DPO_DATA_ROOT/devman; G=$PKG/exp/eval_devman_gt
m=$(slurm/sb slurm/mer.sbatch --parsable --export=ALL,ROUND=$G,WAV_SCP=$D,CANDS_TSV=none,MANIFEST=$M,OUT=$G/mer.tsv)
u=$(slurm/sb slurm/utmos.sbatch --parsable --export=ALL,ROUND=$G,WAV_SCP=$D,CANDS_TSV=none,OUT=$G/utmos.tsv)
c=$(slurm/sb slurm/cmi.sbatch --parsable --export=ALL,ROUND=$G,CANDS_TSV=none,MANIFEST=$M,OUT=$G/cmi.tsv)
slurm/sb slurm/eval.sbatch --dependency=afterok:$m:$u:$c --export=ALL,ROUND=$G,SET_DIR=$D,MER_TSV=$G/mer.tsv,UTMOS_TSV=$G/utmos.tsv,CMI_TSV=$G/cmi.tsv,SET_NAME=devman_gt,OUT=$G/eval.json
```

(ΔCMI is 0 by construction there.) **A synthesized corpus from section 6** (`syn_`-prefixed utts, `wav.scp`
instead of `cands.tsv`) is scored with `WAV_SCP=$O CANDS_TSV=none CAND_LABEL=synth` for mer and utmos
(`MANIFEST=<its manifest>` gives mer its references), with `WAV_SCP=$O CANDS_TSV=none MANIFEST=<its manifest>
CAND_LABEL=synth GT_ALL=1` for cmi (gt + synth rows in one file; `GT_ALL=1` is required because the `syn_`
utts match no gt utt id, so without it 13 would keep no gt row and dCMI would be null), and evaluated
with `SET_DIR=$O CAND_LABEL=synth UTT_PREFIX=syn_` (`eval_tts.py` strips the prefix before the gt CMI join):

```bash
O=$PKG/exp/synth_round1_100h; M=$PKG/data/train_manifest.tsv; E=$PKG/exp/eval_synth_round1
m=$(slurm/sb slurm/mer.sbatch --parsable --export=ALL,ROUND=$E,WAV_SCP=$O,CANDS_TSV=none,MANIFEST=$M,CAND_LABEL=synth,OUT=$E/mer.tsv)
u=$(slurm/sb slurm/utmos.sbatch --parsable --export=ALL,ROUND=$E,WAV_SCP=$O,CANDS_TSV=none,CAND_LABEL=synth,OUT=$E/utmos.tsv)
c=$(slurm/sb slurm/cmi.sbatch --parsable --export=ALL,ROUND=$E,WAV_SCP=$O,CANDS_TSV=none,MANIFEST=$M,CAND_LABEL=synth,GT_ALL=1,OUT=$E/cmi.tsv)
slurm/sb slurm/eval.sbatch --dependency=afterok:$m:$u:$c --export=ALL,ROUND=$E,SET_DIR=$O,MER_TSV=$E/mer.tsv,UTMOS_TSV=$E/utmos.tsv,CMI_TSV=$E/cmi.tsv,CAND_LABEL=synth,UTT_PREFIX=syn_,SET_NAME=synth_round1,OUT=$E/eval.json
```

## 8. Smoke test (sequential, at most 1 GPU, ~15-25 min)

```bash
bash slurm/smoke.sh          # 4 utts of SEAME valid -> $PKG/smoke_out/ ; aborts with "SMOKE FAILED at step <name>"
SMOKE_SFT=1 bash slurm/smoke.sh   # also runs 02_train_sft for 2 steps
```

Each step is a blocking `srun` built from the `CMI_DPO_SLURM_*` variables (see `CLUSTER.md`); logs per
step in `smoke_out/logs/<step>.log`. `smoke_out/` is wiped first unless `SMOKE_KEEP=1`.

## 9. Outputs and file formats

All TSVs are tab-separated with a header row; `cand` is the candidate index (`0..N-1`), `gt`, or the
`CAND_LABEL` of a wav.scp set (`synth`).

| file | producer | columns / content |
|---|---|---|
| `data/<set>_manifest.tsv` (+ `.json`, `.dropped.txt`) | 00 | `utt spk dur wav text_raw text_tts text_ref prompt_utt prompt_wav prompt_text`; json = counts / hours / `hours_cap_selects_all` / prompt counts; dropped = `utt<TAB>reason` |
| `data/sft_cache/<stem>.shard<i>.pt` (+`.json`) | 01 | list of `{utt, text_token int32[L], speech_token int32[T]}`; json = kept/skipped counts |
| `exp/sft_seame/{best.pth,epoch_XXX.pth,train_state.pt,best.json,train.log,args.json}` | 02 | Qwen2LM `state_dict` (no `module.` prefix); `train_state.pt` (~4 GB) = optimizer/scheduler/counters of the checkpoint it names + `epoch_end_done` + `geometry` (world_size/batch/accum/seed), for `RESUME` (a mid-epoch checkpoint is refused under another GPU count / batch / accum / seed; a `best.pth` written on the last step of an epoch resumes into that epoch's pending checkpoint) |
| `<round>/gen/cands.tsv` | 10 | `utt cand wav dur n_tokens` (24 kHz wavs in `gen/wav/<utt>/<k>.wav`) |
| `<round>/gen/tokens/<utt>.pt` | 10 | `{utt, text_ids, text_len_target, prompt_speech, cands:[LongTensor]*N, ended_with_eos:[bool]*N, prompt_utt}` |
| `<round>/gen/gen_config.json`, `<synth>/synth_config.json` | 10 / 20 | config fingerprint (ckpt path/mtime/size, temperature, n_cand, sampling, top_p/top_k, seed, text_frontend, text_mode, `manifest_sha256` = order-independent hash of the manifest's utt/text_raw/text_tts/prompt_utt/prompt_wav/prompt_text columns; synth adds max_dur, keep_truncated, shuffle, sr, utt_prefix); a resume with a different config (incl. a manifest re-generated with other prompts) aborts unless `FORCE_RESUME=1`; a stored fingerprint lacking a newer key only warns and is re-written |
| `<round>/gen/failed.txt`, `<synth>/failed.txt` | 10 / 20 | `utt <TAB> reason`; shards write `failed.txt.<i>`, the `MERGE=1` pass merges them (one line per utt, last wins); 20 also writes `utt <TAB> rejected: <reason>` for truncated / over-length utts and skips those on resume (`--retry_rejected` re-draws them) |
| `<round>/mer.tsv` | 11 | `utt cand wav hyp ref n_ref edits mer` (hyp/ref normalised; corpus MER = Σedits/Σn_ref) |
| `<round>/utmos.tsv` | 12 | `utt cand wav utmos` |
| `<round>/cmi_cands.tsv`, `data/<manifest>.cmi_gt.tsv` (+ `.done`) | 13 | `utt cand wav n_frames n_zh n_en n_blank n_other cmi [text_cmi] [labels_rle]` (cmi in [0,1]; `text_cmi` with `TEXT_CMI=1`, `labels_rle` with `DUMP_LABELS=1`) |
| `<critic out>.failed.txt` | 11 / 12 / 13 | `utt cand wav reason`, rewritten on every run; `RESUME=1` retries exactly these rows |
| `<round>/pairs/pairs.pt` | 14 | list of `{utt, text_ids, prompt_speech, pos, neg, eos_pos, eos_neg, r_pos, r_neg, mer_pos, utmos_pos, dcmi_pos, cand_pos, cand_neg, mer_neg, utmos_neg, dcmi_neg}`; a disabled critic's values may be `None` |
| `<round>/pairs/pairs_summary.json` | 14 | `config` (incl. `mer_clip`, `threshold_disabled_critics`, `max_cand_tokens`, `keep_truncated`), `active_critics`, `pool_filter`, `mer_clip.n_clipped`, kept/`dropped_utts` per reason (`missing_tokens` is tested for every utt of cands.tsv, `tokens_index_missing` also covers utts with >= 2 critic-complete candidates of which < 2 exist in the tokens file, `too_few_after_pool_filter`, `pos_critic_missing`, ...), `candidate_counts` (always present, 0 when nothing counted: `cand_rows`, `cand_scored`, `cand_pooled`, `cand_excluded_over_max_tokens`, `cand_excluded_truncated`, `cand_truncated_kept`, `cand_no_tokens_file`, `cand_index_not_in_tokens`, `cand_missing_<critic>` for active / `cand_missing_<critic>_ignored` for disabled critics; other keys only when non-zero), critic ranges, mean critics pos vs neg |
| `<round>/dpo/{dpo_best.pth,dpo_epoch_XXX.pth,dpo_step_XXXXXX.pth,metrics.jsonl,args.json,val_pairs.txt}` | 15 | same `state_dict` format as SFT (loadable by `cosy.load_llm_ckpt`); `dpo_best.pth` = lowest validation loss on the `VAL_FRAC` held-out pairs listed in `val_pairs.txt` (else lowest epoch training loss); `metrics.jsonl` epoch records carry `val_loss val_margin val_acc n_val best_loss selection`; `train_state.pt` (optimizer + counters + `epoch_end_done` + `geometry` = world_size/batch/accum/seed/val_frac/max_pair_tokens/n_pairs of the writing run) and `ref_cache.pt` (float32 reference log-probs, keyed by `init_ckpt`/`max_pair_tokens`) serve `RESUME=<ckpt>`: a resume is refused when seed/val_frac/max_pair_tokens/n_pairs differ (held-out split would change) or when GPU count/batch/accum differ mid-epoch (another GPU count is accepted from an epoch-boundary `dpo_epoch_XXX.pth`); a `dpo_step` save on the last step of an epoch resumes into that epoch's pending validation/checkpoint; `metrics.jsonl` is truncated by a fresh run |
| `exp/synth_*/{wav/,wav.scp,text,utt2spk,utt2dur}` | 20 | Kaldi dir, `text` = `text_raw`, wav at `SR` (16 kHz), utts prefixed `UTT_PREFIX` |
| `<set>/eval.json` | eval_tts | `utmos_mean, mer_corpus_pct, cmi_mean_pct, cmi_gt_mean_pct, dcmi_mean_pct, ...` |

Every critic script prints one summary line (`corpus_MER% ... failed N`, `mean_UTMOS ... failed N`, the
CMI table) at the end of its log; every sbatch ends with `PREP_DONE / SFT_CACHE_DONE / SFT_DONE / GEN_DONE /
MER_DONE / UTMOS_DONE / CMI_DONE / PAIRS_DONE / DPO_DONE / SYNTH_DONE / EVAL_DONE`; 10/20 end with a
`DONE shard=... done= skipped= failed= [rejected= rejected_skipped= redraws=] ...` line before it (merge
runs: `DONE merge out= ...`).

## 10. Known limitations and explicit choices

**The GPU budget is not enforced by SLURM**: only `run_dpo_round.sh`'s pre-submit check and the
`--dependency` ordering of the recipes above keep this package within the 6-GPU budget of the authors'
cluster (peak counts per recipe in `CLUSTER.md`).

**What the paper leaves open, and the defaults used here** (all overridable):

| choice | default here | note |
|---|---|---|
| candidates per utterance N, temperature τ | `N_CAND=4`, `TEMP=1.0` | RAS sampling (top_p 0.8, top_k 25 from `cosyvoice.yaml`) with logits / τ; min/max token-to-text ratios 2 / 20 as in `Qwen2LM.inference`; candidate k of utt u re-seeded from `sha256(seed, u, k, attempt)` |
| DPO β, lr, epochs, batch | `BETA=0.1`, `LR=1e-6` (constant), `EPOCHS=2`, `BATCH=4` pairs/GPU | AdamW, grad clip 1.0, bf16 policy forward, float32 reference (so with `BF16=1` the step-0 margin is bf16 rounding noise, not exactly 0; the log's `step-0 check` line shows the actual discrepancy) |
| model selection | `VAL_FRAC=0.05` seeded held-out pairs (>= 20 pairs, else none); `dpo_best.pth` = lowest val loss | pairs never trained on; `val_pairs.txt` |
| reward normalisation | `NORM=global` min-max over all pooled candidates of the round (`per_utt` within each utt); zero range maps to 0 | |
| MER clipping | `MER_CLIP=1.0` before normalisation (raw MER is unbounded; a hallucinated candidate at 10-20 would squeeze every other candidate to ~0) | thresholds and `mer_pos/neg` use raw MER; `inf` disables |
| ranking pool | `MAX_CAND_TOKENS=750` (30 s at 25 tok/s) and candidates cut by `max_len` without EOS are excluded (`KEEP_TRUNCATED=1` keeps them) | `MAX_PAIR_TOKENS=750` re-checks at DPO load time |
| gates on the preferred candidate | `MER > 0.20`, `UTMOS < 2.5`, `ΔCMI > 0.20`; applied only for critics with weight > 0 (`THRESHOLD_DISABLED=1` applies all) | ties (`R_pos == R_neg`) dropped; ties in argmax/argmin broken towards the lowest index |
| ΔCMI | `\|CMIspeech(cand) − CMIspeech(gt)\|` (absolute distance) | |
| CMIspeech | `L = {zh, en}`: `T(u)` counts only zh + en frames, so `CMI = min(T_zh, T_en) / (T_zh + T_en) ∈ [0, 0.5]`; `COUNT_ALL_CLASSES=1` includes blank/other (then CMI = 1 − max_k T_k / T over four classes); `T(u) = 0 → CMI = 0` | LAL head classes 0 zh / 1 en / 2 blank / 3 other |
| pseudo labels | encoder-head argmax of the Whisper-LAL `language_cls` (20 ms frames), not decoder cross-attention, at inference | the LAL checkpoint is trained separately (`lal/README.md`) |
| prompt | a different utt of the same speaker, 3-10 s, seeded (zero-shot voice cloning from in-corpus audio); targets without another prompt utt are dropped | `ALLOW_SELF_PROMPT=1` / `PROMPT_MODE=self` restore self-prompting |
| stage-1 batch / schedule | `BATCH=4` per GPU x 4 GPUs (= 16 effective); `GLOBAL_BATCH=4` for the paper literally; warm-up then constant lr (`LR_SCHEDULE=linear` decays) | |
| text | `text_frontend` off: SEAME `text_tts` (space-separated Chinese characters) is tokenised directly with the Qwen tokenizer, as in the authors' stage-1 training; ttsfrd (`TEXT_FRONTEND=1`) would rewrite/split it. `TEXT_MODE=strip` removes `<noise>`/`<unk>`, whereas the authors' stage-1 checkpoint was trained on raw tagged text (`TEXT_MODE=raw` reproduces it; 10/20 take `text_tts`/`prompt_text` verbatim) | |
| synthesis filter | truncated (no EOS) or > `MAX_DUR=30` s generations are re-drawn once (`REJECT_RETRIES=1`) then rejected; rows shuffled with `SEED` | Whisper fine-tuning takes <= 30 s clips |
| "100 h real" | the whole [1, 30] s-filtered train; "100 h synthetic" = one pass over the same transcripts | |

Other facts: sequence log-probs follow the inference layout `[sos, prompt_text ⊕ text, task, prompt_speech ⊕
cand, EOS]` (EOS scored only for candidates that sampled it, no temperature), reference log-probs cached once
in float32 (`CACHE_REF=1`). MER = word-level edit distance over `seame_normalize`d strings (Chinese per
character, English per word), decoded with the HF pipeline (fp16, `chunk_length_s=30`); can exceed 1.0 per
utterance. Speech tokenizer and LAL both see <= 30 s (longer audio is truncated for CMI and skipped in the
SFT cache); candidates are 24 kHz, critics resample to 16 kHz. `sft.sbatch` / `dpo.sbatch` GPU counts are in
the `#SBATCH --gres` header (4 / 2); changing them needs `--gres=gpu:N --export=ALL,GPUS=N`.
`20_synthesize.py --hours` is the corpus-level cap; each shard stops at `hours / nshards`. No downstream ASR
fine-tuning in this repository.

**Runtime and disk planning** (measured on the authors' A40-class GPUs). Roughly 3.3 s per candidate on
~3 s utterances (1 GPU, incl. warm-up), so a full round on the ~89k-utt train manifest (~80k after prompt
drops) x 4 candidates is roughly **300-350 GPU-hours**: use `NSHARDS=5` or `6` (2-3 days per shard,
`gen.sbatch` wall time is 6 days) and **measure on 1k utterances first** (`GEN_LIMIT=1000` in the driver, or
`LIMIT=1000` on a single `gen.sbatch`; `DRY_RUN=1` shows the planned rows). Candidate wavs are 24 kHz
int16: ~**70 GB per round** for ~320k candidates (plus `tokens/`), so delete `gen/wav` of finished rounds.
Scoring ~320k candidates with the MER critic in one 1-GPU job is ~**10 h** (`mer.sbatch` wall time 24 h;
raise it or shard by `LIMIT`/`RESUME` for larger pools); UTMOS and CMI are faster. Synthesis is ~3.3 s per
utterance: 100 h of 16 kHz audio (~90k utts, ~11 GB) is ~80 GPU-hours, 14-16 h per shard with `NSHARDS=6`.
DPO on ~50-70k pairs at `BATCH=4` on 2 GPUs fits the 2-day `dpo.sbatch` wall time; `train_state.pt` and
`ref_cache.pt` add a few GB per experiment.

## 11. Troubleshooting

| symptom | cause / fix |
|---|---|
| job sits in `PENDING (Resources/Priority)` | the partition is shared; `squeue -p $CMI_DPO_SLURM_PARTITION`; lower `NSHARDS`; stay within your GPU budget |
| `run_dpo_round.sh` exits with `WARNING: N GPU(s) already requested` | your other queued jobs (running + pending, dependency-held ones of an earlier round included, because they will run alongside this round) + this round's peak > 6; wait, lower `NSHARDS`/`DPO_GPUS`, or `FORCE=1` when the earlier chain's own peak + this peak fit in the budget |
| `run_dpo_round.sh`: `X must not contain a comma` | paths / EXTRA strings are passed through `--export` (comma-separated); rename the path or put the flag in the sbatch header |
| a dependent job shows `DependencyNeverSatisfied` | an upstream job failed; read `logs/cd_<stage>_<id>.out`, fix, resubmit that stage by hand (section 4; after re-running gen shards also re-run the `MERGE=1` job) and the rest with `--dependency=afterok:<id>` (`scancel` the stuck ones) |
| `10_gen_candidates.py` / `20_synthesize.py` abort with a `gen_config.json` / `synth_config.json` mismatch | you resumed into an `OUT` made with another ckpt / temperature / seed / ... or with a manifest whose prompts/texts differ (`manifest_sha256`; e.g. 00 re-run with another `SEED` / `PROMPT_MODE` / `TEXT_MODE` at the same path); use a new `OUT`, or `FORCE_RESUME=1` to continue anyway (the fingerprint is overwritten and the old and new candidates are mixed). A fingerprint from before a key existed only warns (`predates the key(s) ...`) |
| `02_train_sft.py` / `15_train_dpo.py`: `train_state.pt ... written under another micro-batch partition` / `another pair split` | you resumed with another GPU count / `BATCH` / `ACCUM` (mid-epoch), or, for DPO, another `SEED` / `VAL_FRAC` / `MAX_PAIR_TOKENS` / pairs file (held-out split would change); resume with the original values, from an epoch-boundary checkpoint (`epoch_XXX.pth` / `dpo_epoch_XXX.pth` of a completed epoch: GPU count / batch may change there), or start a new `EXP` |
| `20_synthesize.py` logs `N utts listed as rejected ... are skipped` | expected on resume (deterministic draws); `EXTRA=--retry_rejected` re-draws them, useful only with a changed `REJECT_RETRIES` / `KEEP_TRUNCATED` / `MAX_DUR` (+ `FORCE_RESUME=1`) |
| a critic logs `WARNING ... failed N` but exits 0 | <= `MAX_FAIL_FRAC` (1 %) of the rows were unreadable; they are in `<OUT>.failed.txt`; re-run with `RESUME=1` to retry them, `STRICT=1` to make any failure fatal |
| `11/12/13_score_*.py` exit 2 (`MER_DONE`/`UTMOS_DONE`/`CMI_DONE` missing) | more than `MAX_FAIL_FRAC` of the rows failed (or `STRICT=1`); `<OUT>.failed.txt` lists `utt cand wav reason`; fix the wavs (re-run the gen shard) and re-run with `RESUME=1` |
| `cmi.sbatch`: `at least one of CANDS_TSV / MANIFEST / WAV_SCP is required` | pass `CANDS_TSV=none` (not empty) when scoring gt only, `MANIFEST=<tsv>`; every skip is `VAR=none`, since sbatch drops `VAR=` |
| `cmi.sbatch` crashes unpickling a TTS checkpoint / scores only a fraction of the utts | a `CKPT` / `NSHARDS` from your shell leaked in through `--export=ALL` on a hand-submitted job; the LAL checkpoint is `LAL_CKPT`, and sharding must be passed as `SHARD`/`NSHARDS` on purpose (the driver pins every variable, so this cannot happen through `run_dpo_round.sh`) |
| `13_score_cmi.py`: `No module named WhisperLAL` / unpickling error | `CMI_DPO_LAL_CODE_DIR` must hold `WhisperLAL.py` (default `lal/`); a pickled checkpoint needs the torch version it was saved with, so convert it once with `lal/export_state_dict.py` and point `CMI_DPO_LAL_CKPT` at the `.state_dict.pt` |
| `pairs_summary.json` shows many `missing_gt_cmi` | the gt CMI file is partial (killed / exit-2 job, no `.done` marker); re-run the gt job with `RESUME=1` (the driver does this itself when the marker is missing) |
| `pairs.sbatch`: `header mismatch between ... cmi` | gt and candidate CMI files were made with different `DUMP_LABELS` / `TEXT_CMI`; regenerate one of them |
| `pairs_summary.json` shows `n_pairs: 0` | check `dropped_utts`: `pos_mer_above_max` etc. means thresholds too strict for this model/temperature; `too_few_candidates` means `N_CAND < 2` or critics missing rows (`candidate_counts.cand_missing_*`); `too_few_after_pool_filter` means most candidates are > `MAX_CAND_TOKENS` tokens or truncated (runaway generations: check `cand_excluded_*`, consider `KEEP_TRUNCATED=1` only for diagnosis) |
| `14_build_pairs.py`: `--mer is required while its critic weight is non-zero` | a critic TSV is missing while its weight is > 0; run that critic, or set the weight to 0 |
| `15_train_dpo.py`: `no pairs in ...` | same as above, or every pair exceeded `MAX_PAIR_TOKENS` |
| `15_train_dpo.py` logs `n_val 0` / no `val_loss` | fewer than 20 pairs (smoke) or `VAL_FRAC=0`; `dpo_best.pth` then follows the epoch training loss |
| `02_train_sft.py`: `launch with torchrun (LOCAL_RANK not set)` | run through `sft.sbatch` (uses `torchrun --standalone --nproc_per_node=$GPUS`), not `python` |
| `02_train_sft.py`: `--global_batch ... not divisible` | `GLOBAL_BATCH` must be a multiple of `BATCH x GPUS` (`GLOBAL_BATCH=4` needs `BATCH=1` on 4 GPUs) |
| `02_train_sft.py` / `15_train_dpo.py`: `train_state.pt names another checkpoint` | resume from the checkpoint named in `train_state.pt`, or delete the file for a weights-only resume |
| torchrun `Address already in use` | two torchrun jobs on one node with a fixed port; the sbatch files use `--standalone` (free port), do not add `--master_port` |
| CUDA OOM in DPO | policy + frozen reference + pos/neg stacked forward: lower `BATCH` (raise `ACCUM` / `DPO_ACCUM`), keep `BF16=1`, lower `MAX_PAIR_TOKENS`, or use `--gres=gpu:4 GPUS=4` |
| `01_build_sft_cache.py`: `... exists; pass --overwrite` | on purpose (no clobber); `OVERWRITE=1` |
| `FileExistsError`/`FileNotFoundError` on `--train_cache` glob | quote the glob in `TRAIN_CACHE`; the script expands it itself (`train_manifest.shard*.pt`) |
| `11_score_mer.py --jiwer_check` assertion | corpus MER vs jiwer WER disagree > 1e-6: normalisation bug, report it |
| `HF_HUB_OFFLINE` errors / attempts to download | a model path is wrong, or a hub model (whisper-small, UTMOS) is not cached yet; run once with internet or set the `CMI_DPO_*` paths to local copies |
| `CMI_DPO_<X> is not set` at start-up | fill `config/paths.env` (copy `config/paths.env.example`) or export the variable; `python -c "import cmi_dpo.paths as p; print(p.describe())"` shows what is resolved |
| `sampling_ids` `max_trials` RuntimeError in gen/synth | rare RAS failure; the scripts retry with a fresh seed (`MAX_RETRIES=2` in `gen.sbatch` and `synth.sbatch`, `GEN_MAX_RETRIES` in the driver) and log the utt in `failed.txt` if all attempts fail |
| `20_synthesize.py` lists many `rejected: truncated` / `rejected: too long` | the model babbles past `max_len` for those transcripts; `REJECT_RETRIES=2`, or accept them with `KEEP_TRUNCATED=1` / `MAX_DUR=inf` (they will not fit Whisper's 30 s window) |
| `00_prep_manifest.py` drops many targets (`n_dropped_no_prompt`) | speakers with a single utterance in the pool; `PROMPT_POOL=all` (default) or `ALLOW_SELF_PROMPT=1` |
| WhisperLAL import prints anomaly/CUDA_LAUNCH_BLOCKING notes | `WhisperLAL.py` sets debug flags at import; `lal_cmi` undoes them right after loading |
| smoke test fails at `prep` with `srun: ... queued and waiting` | `srun` is blocking; wait, or run later; every other step is independent enough to resume with `SMOKE_KEEP=1` after fixing |

Logs: `ls -t logs | head`, `tail -f logs/cd_gen0_<id>.out`; job ids of a round: `cat exp/round1/jobs.txt`.
