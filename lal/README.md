# Whisper-LAL (language alignment loss) reference code

Whisper-LAL adds a per-frame language classifier (`language_cls`, a linear layer on the encoder output) to
`openai/whisper-small` and trains it jointly with the ASR objective through a *language alignment loss*: the
decoder's cross-attention maps each transcript token's language (Chinese / English / blank / other) onto the
encoder frames it attends to, and those projected labels supervise the frame classifier (Liu et al., "Aligning
speech to languages to enhance code-switching speech recognition", IEEE TASLP 2025). The resulting model emits
one language label per 20 ms encoder frame, which the CMIspeech critic (`cmi_dpo/lal_cmi.py`,
`scripts/13_score_cmi.py`) turns into the code-mixing index of any wav. `WhisperLAL.py`, `WhisperDataPreLAL.py`,
`train_whisper_LAL.py` and `utils.py` are the authors' training code (verbatim, except that the `--train/--dev/--devman/--devsge`
defaults of `train_whisper_LAL.py` read the environment variable `CMI_DPO_DATA_ROOT`; the script does not load
`config/paths.env` itself, so either export it first (`source slurm/load_env.sh`) or pass the four Kaldi dirs
explicitly as below, otherwise the defaults fall back to the relative `data/SEAME_Segmented/...`); the module name
`WhisperLAL` must not change because the pickled checkpoints reference it.

Training takes Kaldi-style directories (`wav.scp` with 16 kHz audio and `text` with lower-cased transcripts,
Chinese one character per token): `python train_whisper_LAL.py --train <kaldi/train> --dev <kaldi/valid>
--devman <kaldi/devman> --devsge <kaldi/devsge> --model openai/whisper-small --epochs 8 --lr 1e-6 --batch 4
--accumulation 2 --warmup 10000 --lal 0.1 --threshold 0 --module all --layer_index -1 --save_every 10000
--save_dir exp` (`--lal` weights the alignment loss, `--module encoder|decoder|all` chooses what is trained,
`--zeroshot true` evaluates the stock model first). Every `--save_every` steps the dev WER is computed and the
three best checkpoints are kept in `--save_dir` as whole pickled models named
`loss_<loss>_step_<step>_wer_<wer>.pt`; `slurm_train_lal.example` is the one-GPU SLURM script used originally
(env `whisperold`: torch 1.13.1, transformers 4.38.0, openai-whisper, jiwer, langdetect).

Because a pickled module only loads where the training-time torch/transformers versions are importable, export
it once, in that environment, as a portable state dict: `python lal/export_state_dict.py --ckpt
exp/loss_0.201_step_110000_wer_0.6204.pt --out exp/whisper_lal.state_dict.pt [--base_model
openai/whisper-small]`. The output stores the tensors plus `base_model`, `layer_index`, `n_lang` and `d_model`
(`format: cmi_dpo_lal_state_dict_v1`, metadata mirrored in `<out>.json`); point `CMI_DPO_LAL_CKPT` at it and
`load_lal_model` rebuilds `WhisperWithLAL(base_model, layer_index)` from the stored id, or from `CMI_DPO_LAL_BASE_MODEL`
when that variable is set (the base model must be in the HF cache or be a local HF dir when offline; the
variable is how an already exported file is redirected to a local dir) and loads the weights with
`strict=True`, so the scorer runs in the same modern environment as the rest of the pipeline.
