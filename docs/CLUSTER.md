# Cluster layer: SLURM wrapper, GPU accounting, smoke test, legacy environments

Everything GPU-bound in this repository runs as SLURM jobs; the python scripts themselves are plain
CLI programs (`python scripts/<x>.py --help`) and can be run on any machine with the `cstts` env.
This file describes the SLURM layer and the conventions the recipes in [`PIPELINE.md`](PIPELINE.md)
rely on.

## 1. `slurm/sb`: site settings live in `config/paths.env`, not in the sbatch files

No `.sbatch` file contains a partition, node list, account or absolute path. They keep only the
job-independent resource headers (`--job-name`, `--gres`, `--time`, `--cpus-per-task`, `--nodes`,
`--ntasks`) and are always submitted through the wrapper:

```
slurm/sb <file.sbatch> [extra sbatch options...]
```

`sb` derives the repository root `$PKG` from its own location, loads `config/paths.env` through
`slurm/load_env.sh` (every `CMI_DPO_*` line becomes an exported variable, but a `CMI_DPO_*` variable
already set in your shell wins over the file, exactly as in `cmi_dpo/paths.py`; so
`CMI_DPO_ENV_MAIN=other slurm/sb slurm/gen.sbatch` overrides the env name for that job), creates
`$PKG/logs/` and calls

```
sbatch --partition=$CMI_DPO_SLURM_PARTITION [--exclude=$CMI_DPO_SLURM_EXCLUDE] [--account=$CMI_DPO_SLURM_ACCOUNT] \
       $CMI_DPO_SLURM_EXTRA --output=$PKG/logs/%x_%j.out --export=ALL,CMI_DPO_PKG=$PKG  <extra options>  <file.sbatch>
```

`--exclude` / `--account` are added only when the variable is non-empty; `CMI_DPO_SLURM_EXTRA` is
word-split (e.g. `--qos=long --constraint=a40`). Everything after the sbatch file name is passed to
`sbatch`, and sbatch CLI options override `#SBATCH` directives, so `--gres=gpu:0`, `--time`,
`--dependency`, `--parsable`, `--job-name` and `--export=ALL,VAR=value,...` work as usual (always keep
`ALL` in `--export`: it carries the `CMI_DPO_*` variables and `CMI_DPO_PKG` into the job).

Every sbatch body starts the same way:

```bash
PKG=${CMI_DPO_PKG:?run through slurm/sb}
source "$PKG/slurm/load_env.sh"      # config/paths.env; values already in the environment are kept
source "$CMI_DPO_CONDA_SH"; conda activate "$CMI_DPO_ENV_MAIN"      # or ENV_ASR (mer) / ENV_LAL (cmi)
export PYTHONPATH="$CMI_DPO_COSY_ROOT:$CMI_DPO_COSY_ROOT/Matcha-TTS${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$PKG"
```

`HF_HUB_OFFLINE=1` is set because the authors' compute nodes have no internet; on a connected cluster
it is harmless once the hub models (whisper-small, UTMOS) are cached, and you can drop it from the
sbatch files if you prefer downloads inside jobs. Job-specific parameters are `VAR=value` entries in
`--export`; each sbatch header lists them with defaults, and `VAR=none` is the sentinel for "empty"
(sbatch drops `VAR=`). Every job ends with a `<STAGE>_DONE` line; logs are `logs/<job-name>_<jobid>.out`.

Variables read by the SLURM layer (`config/paths.env.example` documents all of them):

| variable | used for |
|---|---|
| `CMI_DPO_SLURM_PARTITION` | `--partition` of every job and of `smoke.sh`'s `srun` |
| `CMI_DPO_SLURM_EXCLUDE` | optional `--exclude` node list (the authors keep faulty / reserved nodes out) |
| `CMI_DPO_SLURM_ACCOUNT` | optional `--account` |
| `CMI_DPO_SLURM_EXTRA` | optional extra options, word-split |
| `CMI_DPO_CONDA_SH` | `conda.sh` sourced inside the job |
| `CMI_DPO_ENV_MAIN`, `CMI_DPO_ENV_ASR`, `CMI_DPO_ENV_LAL` | conda env names (section 5); one env for all three is the intended setup |

## 2. Jobs, resources and the dependency chain

| sbatch | script | env | GPUs | wall time | notes |
|---|---|---|---|---|---|
| `prep.sbatch` | 00_prep_manifest | main | 0 | 1 h | CPU |
| `sft_cache.sbatch` | 01_build_sft_cache | main | 1 | 24 h | shardable (`SHARD`/`NSHARDS`), resumable via `<OUT>.partial` |
| `sft.sbatch` | 02_train_sft (torchrun) | main | 4 | 5 d | `GPUS` must match `--gres` |
| `gen.sbatch` | 10_gen_candidates | main | 1 | 6 d | shardable; `MERGE=1` run with `--gres=gpu:0` |
| `mer.sbatch` | 11_score_mer | asr | 1 | 24 h | |
| `utmos.sbatch` | 12_score_utmos | main | 1 | 24 h | |
| `cmi.sbatch` | 13_score_cmi | lal | 1 | 24 h | writes `<OUT>.done` on success |
| `pairs.sbatch` | 14_build_pairs | main | 0 | 2 h | CPU, deterministic |
| `dpo.sbatch` | 15_train_dpo (torchrun) | main | 2 | 2 d | `GPUS` must match `--gres` |
| `synth.sbatch` | 20_synthesize | main | 1 | 6 d | shardable; `MERGE=1` run with `--gres=gpu:0` |
| `eval.sbatch` | eval_tts | main | 0 | 30 min | CPU |

`slurm/run_dpo_round.sh` submits one DPO round as a chain (`--dependency=afterok`), prints the job
ids and records them in `$OUT/jobs.txt`:

```
[cmi gt  (1 GPU, only when NU != 0 and GT_CMI or its .done marker is missing)]
gen shard 0..NSHARDS-1                       NSHARDS x 1 GPU, concurrently
   └─afterok─► gen merge                     CPU (NSHARDS > 1 only)
         └─afterok─► mer | utmos | cmi       3 x 1 GPU, concurrently
                  └─afterok─► pairs          CPU
                                 └─afterok─► dpo    DPO_GPUS GPUs (torchrun)
```

The driver only calls `slurm/sb`, `squeue` and `awk` (no python on the submit host), pins every
variable of every stage in its `--export` list (so nothing leaks in from the caller's shell), passes
multi-word `*_EXTRA` strings quoted, and refuses `NSHARDS > 6`. `DRY_RUN=1` prints the submit lines
without submitting.

## 3. GPU accounting (the authors' 6-GPU budget)

The authors' cluster allows at most 6 concurrently allocated GPUs per user, and SLURM does not
enforce it for you: the recipes are written so that their own peak stays within it, by chaining
stages with `--dependency` instead of running them side by side. Peak GPU counts:

| recipe (PIPELINE.md) | peak |
|---|---|
| stage 1: token caches then SFT | 5 (caches), then 4 (SFT) |
| one DPO round (= the driver) | `max(NSHARDS + gt, 3 + gt, DPO_GPUS)`, gt = 1 only when a gt-CMI job is submitted (`NU != 0` and no complete `GT_CMI`); it runs beside the gen shards when `NSHARDS + 1 <= 6`, else beside the critics |
| critic ablation (three sequential DPO jobs) | 2 |
| sharded synthesis | `NSHARDS` (<= 6) |
| evaluation of one system | 1 (gen), then 3 (critics) |

`run_dpo_round.sh` adds the GPUs of your running and pending jobs (`squeue`) to its own peak before
submitting and warns / stops above 6 (`FORCE=1` overrides, sensible when an earlier chain's own peak
plus this one fit). Anything you submit by hand on top adds to these numbers:
`squeue -u $USER -o '%i %j %T %b %R'` first. On a cluster without such a limit simply raise `NSHARDS`
(up to the driver's cap of 6 per round) and `DPO_GPUS` / `--gres`.

## 4. `slurm/smoke.sh`: end-to-end test on 4 utterances

```bash
bash slurm/smoke.sh               # ~15-25 min, at most 1 GPU at a time
SMOKE_SFT=1 bash slurm/smoke.sh   # also 2 optimizer steps of 02_train_sft
```

`smoke.sh` derives `$PKG` from its own location, loads `config/paths.env` like `sb`, and runs each
step as a blocking `srun --partition=$CMI_DPO_SLURM_PARTITION [--exclude=...] [--account=...]
$CMI_DPO_SLURM_EXTRA --nodes=1 --ntasks=1 --cpus-per-task=4 --time=$SRUN_TIME [--gres=gpu:1]` in the
right env: 00 prep (CPU), 01 sft cache, [02 sft], 10 gen (2 candidates), 11 mer, 12 utmos, 13 cmi
(gt + candidates), 14 pairs (thresholds relaxed), 15 dpo (torchrun, 1 GPU, 2 steps), 20 synth
(2 utts with the DPO checkpoint) and eval_tts. Parameters: `SMOKE_OUT` (`$PKG/smoke_out`, wiped first
unless `SMOKE_KEEP=1`), `SMOKE_DATA` (`$CMI_DPO_DATA_ROOT/valid`), `SMOKE_SFT`, `SRUN_TIME`
(`00:15:00`). It aborts with `SMOKE FAILED at step <name>` and keeps per-step logs in
`smoke_out/logs/<step>.log`; a run that reaches the end prints `SMOKE_DONE`. The smoke outputs
(`gen/cands.tsv`, `mer.tsv`, `utmos.tsv`, `cmi.tsv`, `eval.json`) are the fixture the authors use
to check that a change leaves the numbers untouched.

The authors' submit host does not allow python at all (not even `python -m py_compile`); hence every
python call, including CPU-only ones, is inside `srun`/`sbatch`, and `run_dpo_round.sh` and `smoke.sh`
are pure bash. Nothing in the package depends on that rule: on a workstation, run the python commands
printed in the smoke log directly.

## 5. Legacy note: why the authors used three conda environments

`env/freeze/` records the three environments the paper's experiments actually ran in:

| env (`CMI_DPO_ENV_*`) | torch / transformers | used for | why separate |
|---|---|---|---|
| `cosyvoicenew` (`ENV_MAIN`) | 2.3.1+cu121 / 4.40.1 | manifests, SFT, candidate generation, UTMOS, pairs, DPO, synthesis, evaluation | the CosyVoice pins |
| `asr-whisper` (`ENV_ASR`) | 2.9.0 / 4.57.1 | MER critic (whisper-large-v3 fine-tuned on SEAME through the HF pipeline; jiwer, opencc) | the critic was trained and validated with a newer transformers; kept identical for the paper |
| `whisperold` (`ENV_LAL`) | 1.13.1+cu117 / 4.38.0 | Whisper-LAL / CMIspeech (`13_score_cmi.py`) | the LAL checkpoint was saved as a **whole pickled model** (`torch.save(model)`) from torch 1.13, and unpickling it needs a compatible torch / transformers |

None of these splits is required by the code:

* `cmi_dpo/common.py` and `cmi_dpo/lal_cmi.py` have no hard dependency beyond torch / numpy /
  openai-whisper (edit distance is implemented locally, opencc is optional), so the MER critic runs
  in the main env as long as its Whisper directory loads with transformers 4.40.1 (a Whisper
  fine-tuned with a newer transformers loads fine; only its `generation_config.json` may need the
  older field names).
* `lal/export_state_dict.py`, run **once** in the environment that can unpickle the checkpoint,
  converts a pickled `WhisperWithLAL` into `<ckpt>.state_dict.pt` plus a small json (base model id,
  layer index, number of classes). `lal_cmi.load_lal_model` accepts either file; with the exported
  state dict the model is rebuilt from the base model id stored in the file (`CMI_DPO_LAL_BASE_MODEL`, when
  set, overrides it, e.g. with a local HF dir on an offline machine) and loaded in the main env.

So a fresh installation sets `CMI_DPO_ENV_MAIN=CMI_DPO_ENV_ASR=CMI_DPO_ENV_LAL=cstts` (the
`env/environment.yml` env) and uses the exported LAL state dict; the freeze files exist only to
reproduce the paper's exact software versions.
