# CSTTS-DPO: code-mixing-guided synthetic speech for code-switching ASR

Code for the TTS side of *Improving Code-Switching ASR with Code-Mixing Guided Synthetic Speech*
(Interspeech 2026, [arXiv:2606.19381](https://arxiv.org/abs/2606.19381)): CosyVoice2 fine-tuning on
SEAME, multi-critic DPO with MER, UTMOS and ΔCMI critics, synthesis of code-switched speech for ASR
augmentation, and the Whisper-LAL model that turns any wav into frame-level language labels and a
speech code-mixing index (CMIspeech).

```bibtex
@inproceedings{yeo2026cmispeech,
  title     = {Improving Code-Switching {ASR} with Code-Mixing Guided Synthetic Speech},
  author    = {Yeo, Yue Heng and Li, Haoyang and Peng, Yizhou and Gopal, Shreyas and Liu, Hexin and Garcia-Perera, Leibny Paola and Sailor, Hardik B. and Wong, Jeremy H. M. and Chng, Eng Siong},
  booktitle = {Proc. Interspeech 2026},
  year      = {2026},
  note      = {arXiv:2606.19381}
}
```

## Setup

```bash
git clone https://github.com/yyheng/CSTTS-DPO.git && cd CSTTS-DPO

# CosyVoice snapshot used in the paper (+ Matcha-TTS, which the fork does not track)
git clone https://github.com/YUCHEN005/TTS_finetune.git third_party/TTS_finetune
git -C third_party/TTS_finetune apply ../../patches/0001-cosyvoice-fork-remove-debug-print.patch
MATCHA=third_party/TTS_finetune/CosyVoice/Matcha-TTS
git clone https://github.com/shivammehta25/Matcha-TTS.git $MATCHA
git -C $MATCHA checkout $(git -C $MATCHA log --reverse --format=%H -S0.0.5.1 -- matcha/VERSION | head -1)

# environment (python 3.10, torch 2.3.1, transformers 4.40.1)
conda env create -f env/environment.yml && conda activate cstts

# CosyVoice2-0.5B
python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"

# machine-specific paths (edit the values)
cp config/paths.env.example config/paths.env
```

`config/paths.env` (all keys prefixed `CMI_DPO_`, absolute paths; `python -c "import cmi_dpo.paths as p; print(p.describe())"` shows the result):

| key | value |
|---|---|
| `COSY_ROOT` | `.../third_party/TTS_finetune/CosyVoice` |
| `COSY_MODEL_DIR` | `.../pretrained_models/CosyVoice2-0.5B` |
| `SFT_CKPT` | stage-1 LLM checkpoint from step 1 (empty = stock CosyVoice2) |
| `LAL_CKPT` | Whisper-LAL checkpoint (step 9) |
| `ASR_MODEL_DIR` | Whisper directory fine-tuned on SEAME (MER critic; any HF Whisper dir works) |
| `UTMOS_SOURCE` | `hub` (downloads UTMOS22 via torch.hub once) or `local` + `UTMOS_REPO` / `UTMOS_CKPT` |
| `DATA_ROOT` | directory holding the SEAME Kaldi dirs `train valid devman devsge` |
| `CONDA_SH`, `ENV_*`, `SLURM_*` | only for the SLURM scripts (`docs/CLUSTER.md`) |

The SEAME-fine-tuned checkpoints (stage-1 LLM, Whisper-LAL, MER critic) are not distributed
(SEAME licence); steps 1 and 9 rebuild them. Everything else is downloaded automatically.

## Usage

Run from the repository root inside the `cstts` env. Every script has `--help`.
On SLURM use `slurm/sb slurm/<step>.sbatch` or `bash slurm/run_dpo_round.sh` (whole DPO round as a
dependency chain); see `docs/CLUSTER.md`. Details of every stage: `docs/PIPELINE.md`.

**0. Manifest** (targets + same-speaker prompts from a Kaldi dir; `--hours H` for a subset)

```bash
python scripts/00_prep_manifest.py --data_dir $CMI_DPO_DATA_ROOT/train --out data/train.tsv
python scripts/00_prep_manifest.py --data_dir $CMI_DPO_DATA_ROOT/valid --out data/valid.tsv
```

**1. Stage 1: fine-tune the CosyVoice2 LLM (SFT)**

```bash
python scripts/01_build_sft_cache.py --manifest data/train.tsv --out data/sft_cache/train.pt   # --shard i --nshards n for parallel jobs
python scripts/01_build_sft_cache.py --manifest data/valid.tsv --out data/sft_cache/valid.pt
torchrun --standalone --nproc_per_node=4 scripts/02_train_sft.py \
    --train_cache 'data/sft_cache/train*.pt' --valid_cache data/sft_cache/valid.pt --exp exp/sft \
    --lr 2e-4 --batch 1 --global_batch 4 --max_steps 50000          # -> exp/sft/best.pth  (set CMI_DPO_SFT_CKPT to it)
```

**2. Candidates** (N samples per transcript at temperature τ)

```bash
python scripts/10_gen_candidates.py --manifest data/train.tsv --out exp/round1/gen --n_cand 4 --temperature 1.0 --seed 0
# parallel: --shard i --nshards n per job, then:  python scripts/10_gen_candidates.py --out exp/round1/gen --merge
```

**3. Critics** (MER, UTMOS, CMIspeech of candidates and of the real recordings)

```bash
R=exp/round1
python scripts/11_score_mer.py   --cands_tsv $R/gen/cands.tsv --manifest data/train.tsv --out $R/mer.tsv
python scripts/12_score_utmos.py --cands_tsv $R/gen/cands.tsv --out $R/utmos.tsv
python scripts/13_score_cmi.py   --cands_tsv $R/gen/cands.tsv --manifest data/train.tsv --out $R/cmi.tsv --print_summary
```

**4. Preference pairs** (`R = λ·UTMOS − γ·MER − ν·ΔCMI`, best vs. worst candidate, thresholds on the preferred one)

```bash
python scripts/14_build_pairs.py --cands $R/gen/cands.tsv --mer $R/mer.tsv --utmos $R/utmos.tsv --cmi $R/cmi.tsv \
    --tokens_dir $R/gen/tokens --out $R/pairs --lam 1 --gam 1 --nu 1
# ablations: --lam 0 --nu 0 (MER only), --nu 0 (MER + UTMOS); a weight of 0 disables that critic and its threshold
```

**5. Stage 2: DPO**

```bash
torchrun --standalone --nproc_per_node=2 scripts/15_train_dpo.py --pairs $R/pairs/pairs.pt --exp $R/dpo \
    --init_ckpt $CMI_DPO_SFT_CKPT --beta 0.1 --lr 1e-6 --epochs 2 --batch 4 --bf16   # -> exp/round1/dpo/dpo_best.pth
```

**6. Synthesize an augmentation corpus** (16 kHz Kaldi dir: `wav/ wav.scp text utt2spk utt2dur`)

```bash
python scripts/20_synthesize.py --manifest data/train.tsv --ckpt $R/dpo/dpo_best.pth --out exp/synth --utt_prefix syn_ --hours 100
# parallel: --shard i --nshards n per job (same --hours), then:  python scripts/20_synthesize.py --out exp/synth --merge
```

**7. Evaluate a TTS model** (UTMOS, MER, ΔCMI on devman / devsge)

```bash
python scripts/00_prep_manifest.py --data_dir $CMI_DPO_DATA_ROOT/devman --out data/devman.tsv
S=exp/eval_devman; M=data/devman.tsv
python scripts/10_gen_candidates.py --manifest $M --out $S/gen --ckpt $R/dpo/dpo_best.pth --n_cand 1
python scripts/11_score_mer.py   --cands_tsv $S/gen/cands.tsv --manifest $M --out $S/mer.tsv
python scripts/12_score_utmos.py --cands_tsv $S/gen/cands.tsv --out $S/utmos.tsv
python scripts/13_score_cmi.py   --cands_tsv $S/gen/cands.tsv --manifest $M --out $S/cmi.tsv
python scripts/eval_tts.py --set_dir $S/gen --mer_tsv $S/mer.tsv --utmos_tsv $S/utmos.tsv --cmi_tsv $S/cmi.tsv --set_name devman   # -> $S/gen/eval.json
```

**8. One sentence** (zero-shot voice cloning from a 3-10 s prompt; SEAME-style text = one Chinese character per token)

```bash
python scripts/infer_one.py --text "我 觉 得 the project 还 可 以" --prompt_wav /path/to/prompt.wav \
    --prompt_text "prompt transcript" --ckpt $R/dpo/dpo_best.pth --out out.wav      # --ckpt none = stock CosyVoice2, --sr 16000, --n 3
```

**9. Whisper-LAL: train, export, score any wavs**

```bash
python lal/train_whisper_LAL.py --train $CMI_DPO_DATA_ROOT/train --dev $CMI_DPO_DATA_ROOT/valid \
    --model openai/whisper-small --save_dir exp/lal                               # -> exp/lal/<best>.pt
python lal/export_state_dict.py --ckpt exp/lal/<best>.pt --out exp/lal/lal.state_dict.pt   # set CMI_DPO_LAL_CKPT to it
python scripts/13_score_cmi.py --wav_scp /path/to/kaldi_dir --out cmi.tsv --print_summary   # utt cand wav n_frames n_zh n_en n_blank n_other cmi
```

## License

Apache-2.0 (`LICENSE`). Built on [CosyVoice](https://github.com/FunAudioLLM/CosyVoice),
[Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS), [SpeechMOS / UTMOS](https://github.com/tarepan/SpeechMOS)
and [Whisper](https://github.com/openai/whisper); see `NOTICE`. The SEAME corpus and models fine-tuned
on it are covered by the SEAME licence.
