# CSTTS-DPO: code-mixing-guided synthetic speech for code-switching ASR

Reference implementation of the TTS side of *Improving Code-Switching ASR with Code-Mixing Guided
Synthetic Speech* (arXiv:2606.19381, 2026): a CosyVoice2 speech-token LLM is fine-tuned on
Mandarin-English SEAME, then aligned with multi-critic Direct Preference Optimization whose critics
are MER (intelligibility, from a SEAME-fine-tuned Whisper), UTMOS (naturalness) and ΔCMI (the
distance between the code-mixing index of the synthetic speech and that of the real recording,
measured on frame-level language labels from a Whisper-LAL model). The aligned model synthesizes
code-switched speech for ASR data augmentation. Downstream ASR fine-tuning is out of scope of this
repository.

## Paper

Yeo, Li, Peng, Gopal, Liu, Garcia-Perera, Sailor, Wong and Chng, *Improving Code-Switching ASR with
Code-Mixing Guided Synthetic Speech*, Proc. Interspeech 2026 (accepted). arXiv:2606.19381,
<https://arxiv.org/abs/2606.19381>.

```bibtex
@inproceedings{yeo2026cmispeech,
  title     = {Improving Code-Switching {ASR} with Code-Mixing Guided Synthetic Speech},
  author    = {Yeo, Yue Heng and Li, Haoyang and Peng, Yizhou and Gopal, Shreyas and Liu, Hexin and Garcia-Perera, Leibny Paola and Sailor, Hardik B. and Wong, Jeremy H. M. and Chng, Eng Siong},
  booktitle = {Proc. Interspeech 2026},
  year      = {2026},
  note      = {arXiv:2606.19381}
}
```

## What is in this repository

**Component 1: TTS.** Stage 1 fine-tunes the CosyVoice2-0.5B speech-token LLM on SEAME (SFT).
Stage 2 samples several candidates per training transcript, scores each with the three critics,
builds preference pairs (best vs. worst candidate under the weighted reward
`R = λ·UTMOS − γ·MER − ν·ΔCMI`) and runs DPO. The resulting model synthesizes a code-switched
augmentation corpus (one utterance per transcript, zero-shot voice cloning from a same-speaker
prompt).

**Component 2: Whisper-LAL / CMIspeech.** A whisper-small encoder with a frame-level language head
(20 ms frames, classes zh / en / blank / other) produces pseudo language labels for any wav, from
which the speech code-mixing index `CMIspeech(u) = (T − max_k T_k) / T` over zh+en frames and
`ΔCMI = |CMIspeech(synth) − CMIspeech(real)|` are computed. It works on real and synthetic audio and
is usable on its own (`scripts/13_score_cmi.py`).

```
 SEAME Kaldi dir ──► 00 manifest ──► 01 token cache ──► 02 SFT (stage 1) ──► sft ckpt
                          │                                                     │
                          │            ┌────────────────────────────────────────┘
                          ▼            ▼
                     10 gen N candidates / utt ──► 11 MER ─┐
                                                   12 UTMOS ├─► 14 pairs (R = λU − γM − νΔCMI) ──► 15 DPO (stage 2) ──► dpo ckpt
                     real wavs ──► 13 CMI (gt) ──► 13 CMI ─┘                                                            │
                                   (Whisper-LAL)                                                                          ▼
                                                                20 synthesize augmentation corpus ◄─── (or infer_one.py for one sentence)
                                                                                    │
                                                                 11 / 12 / 13 + eval_tts.py (UTMOS, MER, ΔCMI)
```

Layout:

```
cmi_dpo/          common.py (manifest / Kaldi IO, SEAME normaliser, MER)  cosy.py (CosyVoice2 glue:
                  prompt inputs, sampling with temperature, token-to-wav, sequence log-probs for DPO)
                  lal_cmi.py (Whisper-LAL labels, CMI)  paths.py (configuration, see below)
scripts/          00_prep_manifest.py 01_build_sft_cache.py 02_train_sft.py 10_gen_candidates.py
                  11_score_mer.py 12_score_utmos.py 13_score_cmi.py 14_build_pairs.py 15_train_dpo.py
                  20_synthesize.py eval_tts.py infer_one.py
lal/              Whisper-LAL model, data preparation and training code + export_state_dict.py
slurm/            sb (sbatch wrapper) + one .sbatch per script + run_dpo_round.sh + smoke.sh
config/           paths.env.example (copy to paths.env)
env/              environment.yml, requirements.txt, freeze/ (the exact environments used)
docs/             PIPELINE.md (training and evaluation recipes), CLUSTER.md (SLURM layer), DESIGN.md (design
                  notes: data conventions, CosyVoice2 and Whisper-LAL facts the code relies on, function reference)
patches/          one-line patch for the CosyVoice fork
```

## Installation

1. Clone this repository and the CosyVoice snapshot used in the paper (the authors' fork of
   CosyVoice). The fork does **not** track `Matcha-TTS` (CosyVoice imports it as a submodule at
   `third_party/Matcha-TTS`; in the snapshot it lives untracked at `CosyVoice/Matcha-TTS`), so clone
   it separately into that location and pin it: the paper's runs used Matcha-TTS 0.0.5.1 (the
   authors' copy carries no git metadata, so the checkout below selects the upstream commit that set
   `matcha/VERSION` to 0.0.5.1; `cosyvoice.flow` imports `matcha.models.components.*` and
   `matcha.hifigan.models`, which later upstream commits may change).

   ```bash
   git clone https://github.com/yyheng/CSTTS-DPO.git && cd CSTTS-DPO
   git clone https://github.com/YUCHEN005/TTS_finetune.git third_party/TTS_finetune
   git -C third_party/TTS_finetune apply ../../patches/0001-cosyvoice-fork-remove-debug-print.patch
   MATCHA=third_party/TTS_finetune/CosyVoice/Matcha-TTS
   git clone https://github.com/shivammehta25/Matcha-TTS.git $MATCHA
   git -C $MATCHA checkout $(git -C $MATCHA log --reverse --format=%H -S0.0.5.1 -- matcha/VERSION | head -1)
   cat $MATCHA/matcha/VERSION    # 0.0.5.1
   ```

   `cmi_dpo/cosy.py` puts `$CMI_DPO_COSY_ROOT` and `$CMI_DPO_COSY_ROOT/Matcha-TTS` on `sys.path`
   itself, so no `PYTHONPATH` is needed for the python scripts (the sbatch files export it anyway).

2. Python environment (python 3.10, torch 2.3.1 / cu121, transformers 4.40.1, the pins of the
   CosyVoice snapshot):

   ```bash
   conda env create -f env/environment.yml      # creates env "cstts" (pynini from conda-forge)
   conda activate cstts
   pip install -r env/requirements.txt          # already run by the yml; re-run after edits
   ```

   `env/freeze/` holds the exact `pip freeze` of the three environments the paper's numbers were
   produced with; `docs/CLUSTER.md` explains why there were three and why one is enough now.

3. Pretrained models.

   * **CosyVoice2-0.5B** (the CosyVoice README's command; needs `modelscope`, which is in the
     requirements):

     ```python
     from modelscope import snapshot_download
     snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
     ```

     CosyVoice's optional `ttsfrd` text front end is not needed: without the `ttsfrd` wheel the
     fork uses WeTextProcessing (in the requirements), which is what this pipeline runs with, and
     SEAME text is fed to the tokenizer verbatim anyway (a text normaliser is only applied with
     `--text_frontend`). If you do install the wheel, CosyVoice initialises it at load time from
     `$CMI_DPO_COSY_ROOT/pretrained_models/CosyVoice-ttsfrd/resource` (download `iic/CosyVoice-ttsfrd`
     there, not into the repository root).
   * **whisper-large-v3** (<https://huggingface.co/openai/whisper-large-v3>) is the base of the MER
     critic. The critic itself is whisper-large-v3 fine-tuned on SEAME (any Hugging Face Whisper
     directory usable with `transformers.pipeline('automatic-speech-recognition')` works).
   * **UTMOS** (UTMOS22 strong, SpeechMOS) is loaded through `torch.hub` on first use with
     `CMI_DPO_UTMOS_SOURCE=hub` (internet needed once; the hub cache is reused afterwards). On an
     offline machine set `CMI_DPO_UTMOS_SOURCE=local` and point `CMI_DPO_UTMOS_REPO` /
     `CMI_DPO_UTMOS_CKPT` to a local clone of SpeechMOS and its `utmos22_strong` checkpoint.
   * **whisper-small** (<https://huggingface.co/openai/whisper-small>) is the base of Whisper-LAL
     (`CMI_DPO_LAL_BASE_MODEL`, downloaded by `transformers` on first use; offline, point the variable
     at a local copy of the model directory).

   **Not distributed:** the SEAME-fine-tuned checkpoints (the stage-1 CosyVoice2 LLM, the Whisper-LAL
   language head and the MER critic) are derived from the SEAME corpus, whose licence does not allow
   redistribution. With a SEAME licence you can rebuild them: the stage-1 LLM with `docs/PIPELINE.md`
   (stage 1) and the Whisper-LAL model with `lal/README.md` (`lal/train_whisper_LAL.py`). The MER
   critic is not covered by this repository: it is any Hugging Face Whisper directory fine-tuned on
   the SEAME train set with standard tooling (the authors fully fine-tuned `openai/whisper-large-v3`
   with the Hugging Face `transformers` Trainer), set as `CMI_DPO_ASR_MODEL_DIR`. Without them the
   stock CosyVoice2 model still runs end to end (`--ckpt none` / empty `CMI_DPO_SFT_CKPT`), UTMOS works
   as is, and only the MER and ΔCMI critics need a replacement.

## Configuration

All machine-specific locations come from environment variables with the prefix `CMI_DPO_`, read by
`cmi_dpo/paths.py`. The loader also reads `config/paths.env` (plain `KEY=VALUE` lines, `#` comments,
an optional `export` prefix; variables already set in the shell win over the file, in python and in
the bash layer alike; paths must be absolute) from the repository root, so the usual setup is:

```bash
cp config/paths.env.example config/paths.env   # git-ignored; edit the values
python -c "import cmi_dpo.paths as p; print(p.describe())"   # shows every key and its current value
```

| variable | meaning |
|---|---|
| `CMI_DPO_COSY_ROOT` | CosyVoice code directory that contains `cosyvoice/` and `Matcha-TTS/`; an **absolute** path (values are not resolved against the repository root), e.g. `/path/to/CSTTS-DPO/third_party/TTS_finetune/CosyVoice` after the clone above |
| `CMI_DPO_COSY_MODEL_DIR` | CosyVoice2-0.5B model directory (`cosyvoice.yaml`, `llm.pt`, `flow.pt`, `hift.pt`, `CosyVoice-BlankEN`) |
| `CMI_DPO_SFT_CKPT` | optional stage-1 LLM checkpoint (Qwen2LM state dict); empty = the stock `llm.pt` |
| `CMI_DPO_LAL_CODE_DIR` | directory holding `WhisperLAL.py` (default: `lal/` of this repository) |
| `CMI_DPO_LAL_CKPT` | Whisper-LAL checkpoint: a pickled `WhisperWithLAL` or the `.state_dict.pt` written by `lal/export_state_dict.py` |
| `CMI_DPO_LAL_BASE_MODEL` | Hugging Face id or local directory of the LAL base model (default `openai/whisper-small`); when set it overrides the id stored in an exported state dict |
| `CMI_DPO_ASR_MODEL_DIR` | Hugging Face Whisper directory of the MER critic |
| `CMI_DPO_UTMOS_SOURCE` | `hub` (torch.hub, default) or `local` |
| `CMI_DPO_UTMOS_REPO` | local SpeechMOS clone (only with `CMI_DPO_UTMOS_SOURCE=local`) |
| `CMI_DPO_UTMOS_CKPT` | local `utmos22_strong` state dict (only with `CMI_DPO_UTMOS_SOURCE=local`) |
| `CMI_DPO_DATA_ROOT` | directory with the SEAME Kaldi dirs `train valid devman devsge` (`wav.scp text utt2spk utt2dur`) |
| `CMI_DPO_CONDA_SH` | `conda.sh` sourced by the SLURM jobs |
| `CMI_DPO_ENV_MAIN` / `CMI_DPO_ENV_ASR` / `CMI_DPO_ENV_LAL` | conda env names used by the sbatch files (all three may be the same env, e.g. `cstts`) |
| `CMI_DPO_SLURM_PARTITION` | partition passed by `slurm/sb` |
| `CMI_DPO_SLURM_EXCLUDE` | optional `--exclude` node list |
| `CMI_DPO_SLURM_ACCOUNT` | optional `--account` |
| `CMI_DPO_SLURM_EXTRA` | optional extra `sbatch` options (word-split) |

Every script also accepts the corresponding CLI flags (`--ckpt`, `--model_dir`, `--utmos_source`, `--utmos_repo`, ...), which
override the variables; `--show_paths` prints the resolved configuration. A missing required
variable raises a clear error naming it and `config/paths.env.example`.

## Inference

Run the commands from the repository root inside the `cstts` env with `config/paths.env` filled in.
Every script has `--help`. The examples use placeholder paths.

**a. One sentence** (zero-shot voice cloning from a 3-10 s prompt of the target speaker; the prompt
transcript conditions the LLM, so it must be exact). SEAME-style input text is space-separated with
one Chinese character per token, e.g. `"我 觉 得 the project 还 可 以"`:

```bash
python scripts/infer_one.py \
    --text "我 觉 得 the project 还 可 以" \
    --prompt_wav /path/to/prompt.wav --prompt_text "prompt transcript in the same convention" \
    --ckpt /path/to/dpo_best.pth --out out.wav --temperature 1.0 --seed 0
# --ckpt none = stock CosyVoice2; --sr 16000 writes 16 kHz instead of 24 kHz; --n 3 samples three wavs
```

**b. Batch synthesis of a manifest** (one utterance per transcript, same-speaker prompts drawn
from the Kaldi dir, 16 kHz Kaldi-style output ready for ASR training):

```bash
python scripts/00_prep_manifest.py --data_dir /path/to/kaldi_dir --out data/my_manifest.tsv
python scripts/20_synthesize.py --manifest data/my_manifest.tsv --ckpt /path/to/dpo_best.pth \
    --out exp/synth_my --utt_prefix syn_ --temperature 1.0 --seed 0
# -> exp/synth_my/{wav/,wav.scp,text,utt2spk,utt2dur,failed.txt,synth_config.json}
# several GPUs: --shard i --nshards n per job, then once:  python scripts/20_synthesize.py --out exp/synth_my --merge
# --hours H caps the corpus; --dry_run prints the plan; a re-run resumes where it stopped
```

**c. CMIspeech / ΔCMI of any wav list** (Whisper-LAL; needs `CMI_DPO_LAL_CKPT`):

```bash
# wav.scp mode: a Kaldi dir (or its wav.scp) -> one row per wav
python scripts/13_score_cmi.py --wav_scp /path/to/kaldi_dir --out cmi.tsv --print_summary
# cands.tsv mode: candidate rows (utt cand wav ...) + the real recordings of --manifest (rows cand=gt)
python scripts/13_score_cmi.py --cands_tsv exp/round1/gen/cands.tsv --manifest data/my_manifest.tsv \
    --out cmi.tsv --print_summary
# synthetic corpus vs. its real counterpart: --wav_scp exp/synth_my --manifest data/my_manifest.tsv --cand_label synth --gt_all
```

Output TSV columns: `utt cand wav n_frames n_zh n_en n_blank n_other cmi` (`cand` = `gt` for real
recordings, the candidate index or `--cand_label` otherwise; `n_*` = frames per class; `cmi` in
[0, 1]), plus `text_cmi` with `--text_cmi` (token-level CMI of the manifest text) and `labels_rle`
with `--dump_labels` (run-length-encoded frame labels, e.g. `0x120,1x35,2x10`). `--print_summary`
prints the mean CMI per `cand` and the mean ΔCMI against the `gt` rows in percent.

**d. Scoring a synthesized set with the three critics** (MER needs `CMI_DPO_ASR_MODEL_DIR`, UTMOS
needs internet once or `CMI_DPO_UTMOS_SOURCE=local`):

```bash
S=exp/synth_my; M=data/my_manifest.tsv
python scripts/11_score_mer.py   --wav_scp $S --manifest $M --cand_label synth --out $S/mer.tsv
python scripts/12_score_utmos.py --wav_scp $S --cand_label synth --out $S/utmos.tsv
python scripts/13_score_cmi.py   --wav_scp $S --manifest $M --cand_label synth --gt_all --out $S/cmi.tsv
python scripts/eval_tts.py --set_dir $S --mer_tsv $S/mer.tsv --utmos_tsv $S/utmos.tsv --cmi_tsv $S/cmi.tsv \
    --cand_label synth --utt_prefix syn_ --set_name synth_my
# -> $S/eval.json: utmos_mean, mer_corpus_pct, cmi_mean_pct, cmi_gt_mean_pct, dcmi_mean_pct, ...
```

For a candidate set produced by `10_gen_candidates.py` use `--cands_tsv <dir>/cands.tsv` instead of
`--wav_scp` in all three critics and `--set_dir <dir>` (no `--cand_label` / `--utt_prefix`) in
`eval_tts.py`.

## Training

* `docs/PIPELINE.md`: data conventions, stage 1 (SFT token cache + DDP fine-tuning), stage 2 (one
  DPO round: candidates, critics, pairs, DPO), the critic ablation by weights, sharded synthesis,
  evaluation, output formats, known choices and troubleshooting.
* `docs/CLUSTER.md`: the SLURM layer (`slurm/sb`, the sbatch files, `run_dpo_round.sh` dependency
  chain, GPU accounting, `smoke.sh`) and the authors' legacy three-environment layout.
* `lal/README.md`: training the Whisper-LAL language head and exporting its state dict.

## License

This repository is released under the Apache License 2.0 (`LICENSE`). CosyVoice is Apache-2.0,
Matcha-TTS, SpeechMOS/UTMOS and OpenAI Whisper are MIT; see `NOTICE`. The SEAME corpus and all models
fine-tuned on it are covered by the SEAME licence, which is separate from this code licence.

## Acknowledgements

Built on [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) (CosyVoice2-0.5B and its training
code), [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS),
[SpeechMOS / UTMOS](https://github.com/tarepan/SpeechMOS), [Whisper](https://github.com/openai/whisper)
and Hugging Face `transformers`, and on the SEAME corpus (LDC2015S04).

## Citation

If you use this code, please cite the paper (see `CITATION.cff` and the BibTeX entry above):
Yeo et al., *Improving Code-Switching ASR with Code-Mixing Guided Synthetic Speech*, Proc. Interspeech
2026 (arXiv:2606.19381).
