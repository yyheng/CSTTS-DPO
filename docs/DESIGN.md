# cmi_dpo — Design notes

Implementation of arXiv 2606.19381 "Improving Code-Switching ASR with Code-Mixing Guided
Synthetic Speech" (Yeo et al., Proc. Interspeech 2026, arXiv:2606.19381; venue per the arXiv listing
"Accepted to Interspeech 2026"), restricted to:
  (A) TTS side: stage-1 basic fine-tuning of CosyVoice2 on SEAME, stage-2 multi-critic DPO
      (critics = MER, UTMOS, ΔCMI), and inference/synthesis of CS speech for augmentation.
  (B) The Whisper-LAL component that produces pseudo frame-level language labels and
      CMIspeech / ΔCMI for any wav list (ground truth or synthetic).
Downstream ASR fine-tuning is out of scope.

These notes collect the facts about the external components (CosyVoice2, Whisper-LAL, UTMOS, the MER
critic), the data conventions and the file formats that the code relies on, together with the
library API of the `cmi_dpo` package. They are written for maintainers; the recipes for running the pipeline are
in README.md, docs/PIPELINE.md and docs/CLUSTER.md. Where a statement disagrees with the code, the
code is authoritative.

## 0. Cluster conventions (see docs/CLUSTER.md)
All site-specific values come from `config/paths.env` (variables `CMI_DPO_*`, loaded by
`cmi_dpo/paths.py` in python and by `slurm/load_env.sh` in bash, sourced by `slurm/sb`, `smoke.sh`,
`run_dpo_round.sh` and every sbatch body; in both layers a variable already set in the environment
wins over the file; paths in the file are absolute); no tracked file contains an absolute path of a
particular machine. On the authors' cluster the submit (login) host runs no python at all (every
invocation, syntax check or import check goes through `srun`/`sbatch`; CPU-only checks use
`srun --partition=$CMI_DPO_SLURM_PARTITION --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:10:00 bash -lc '...'`),
at most 6 GPUs are in use concurrently per user (jobs are chained with `--dependency=afterok:<id>`
rather than run in parallel), the partition / node exclusions / account are not written into sbatch
headers (`slurm/sb` adds `--partition=$CMI_DPO_SLURM_PARTITION [--exclude=$CMI_DPO_SLURM_EXCLUDE]
[--account=...] $CMI_DPO_SLURM_EXTRA --output=$PKG/logs/%x_%j.out --export=ALL,CMI_DPO_PKG=$PKG` at
submit time), compute nodes are offline (`export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`; nothing
is downloaded inside a job, all models/checkpoints are local), conda is activated with
`source "$CMI_DPO_CONDA_SH" && conda activate "$CMI_DPO_ENV_MAIN"` (or `ENV_ASR` / `ENV_LAL`), logs
go to `$PKG/logs/%x_%j.out` (`sb` creates the dir) and every job body uses `set -euo pipefail` and
`python -u`. The CosyVoice fork, the original Whisper-LAL code, the critic checkpoints and the data
are read-only references outside the repository.

## 1. Environments (conda, python 3.10) and what runs where
| env          | torch            | transformers | extras                                   | used for |
|--------------|------------------|--------------|------------------------------------------|----------|
| cosyvoicenew | 2.3.1+cu121      | 4.40.1       | torchaudio 2.3.1, openai-whisper, onnxruntime, hyperpyyaml, librosa, soundfile, pandas, tqdm; NO jiwer, NO opencc | manifest/cache prep, SFT, candidate generation, UTMOS, pair building, DPO, synthesis |
| asr-whisper  | 2.9.0            | 4.57.1       | jiwer 4.0, opencc, librosa, accelerate   | MER critic (Whisper-large-v3 SEAME FT) |
| whisperold   | 1.13.1+cu117     | 4.38.0       | openai-whisper 20250625, torchaudio 0.13.1, jiwer, langdetect | Whisper-LAL / CMIspeech component (the LAL checkpoint was trained+pickled in this env) |

The pure-python modules shared across environments (common.py, lal_cmi.py) do not import
jiwer/opencc/pandas unconditionally: the edit distance is implemented in common.py and opencc is
optional (try/except). Type hints are kept 3.10-compatible (`X | None` inside runtime-evaluated
positions is fine in 3.10; 3.11-only features are avoided).

## 2. Data (Kaldi dirs, all 16 kHz mono flac)
$CMI_DPO_DATA_ROOT/{train,valid,devman,devsge}/ each with
  wav.scp   "<utt> <abs path .flac>"
  text      "<utt> tok tok ..." lowercase; Chinese already ONE CHARACTER PER TOKEN (space-separated);
            English words; may contain tags like <noise> <unk>
  utt2spk   "<utt> <spk>"      utt2dur "<utt> <seconds>"     spk2utt
Sizes: train 89,339 utts / 96.35 h / 134 speakers; valid 4,725 utts / 5.11 h; devman 6,531; devsge 5,321.
Utterance ids look like nc01f-01nc01fbx_0101-000000-000601 (speaker prefix before '-').

Text conventions used by this package:
  text_raw = transcript as in `text` (tokens joined by single spaces)
  text_tts = TTS conditioning text. 00_prep_manifest --text_mode strip (default): text_raw with
             <...> tags removed, whitespace collapsed; --text_mode raw: text_raw verbatim (tags kept).
             The authors' stage-1 model (SEAME_Final/epoch_020.pth, the CMI_DPO_SFT_CKPT on their cluster)
             was trained on the RAW text incl. <noise>/<UNK> (TTS_finetune/CosyVoice/data_processing.py
             passes the `text` line straight to frontend.data_processing -> tokenizer.encode; train
             text has 7,644 <noise> + 545 <UNK>, some glued like "<noise>then"). To reproduce that
             convention (01/02 SFT, or candidate generation matched to epoch_020) use --text_mode raw.
             Either way the SEAME spacing is kept (Chinese chars space-separated). prompt_text follows
             the same mode. 10_gen_candidates/20_synthesize take text_tts AND prompt_text verbatim (no re-strip).
  text_ref = seame_normalize(text_raw)  (see §6 MER) used as the MER reference. seame_normalize is
             env-dependent (opencc t2s only where opencc is importable = asr-whisper). 00_prep_manifest
             therefore writes text_ref ONLY when opencc is importable; in cosyvoicenew the column is
             EMPTY and 11_score_mer.load_refs recomputes seame_normalize(text_raw) in the scoring env
             (its empty-ref fallback). The manifest's text_ref column is advisory.

## 3. CosyVoice2 (repo $CMI_DPO_COSY_ROOT = the authors' CosyVoice fork + Matcha-TTS, env cosyvoicenew)
Constants (all resolved by cmi_dpo/paths.py from the CMI_DPO_* variables, see §14):
  COSY_ROOT      = $CMI_DPO_COSY_ROOT        (CosyVoice code dir containing cosyvoice/ and Matcha-TTS/)
  COSY_MODEL_DIR = $CMI_DPO_COSY_MODEL_DIR   (CosyVoice2-0.5B: cosyvoice.yaml llm.pt flow.pt hift.pt CosyVoice-BlankEN/)
  SFT_LLM_CKPT   = $CMI_DPO_SFT_CKPT         (optional; '' = stock llm.pt. The authors' stage-1 model is a
                   DDP state_dict of Qwen2LM with "module." prefix, trained on SEAME; not distributed)
  Import setup: sys.path.insert(0, COSY_ROOT); sys.path.insert(0, COSY_ROOT + '/Matcha-TTS')
  ttsfrd (only if its wheel is installed; otherwise WeTextProcessing, the case of every run reported here)
  is initialised by CosyVoiceFrontEnd from COSY_ROOT/pretrained_models/CosyVoice-ttsfrd/resource, resolved
  relative to cosyvoice/cli/frontend.py, so cwd does not matter; model paths are always passed absolute.

Construction:
  from cosyvoice.cli.cosyvoice import CosyVoice2
  cv = CosyVoice2(COSY_MODEL_DIR, load_jit=False, load_trt=False, fp16=False, device=torch.device('cuda:0'))
  cv.sample_rate == 24000 ; cv.frontend : CosyVoiceFrontEnd ; cv.model : CosyVoice2Model
  cv.model.llm : Qwen2LM ; cv.model.flow ; cv.model.hift
  Loading a fine-tuned LLM state dict (works for both stage-1 and DPO checkpoints):
     sd = torch.load(ckpt, map_location='cpu'); sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
     cv.model.llm.load_state_dict(sd, strict=True)
  (CosyVoice2.__init__ reads {model_dir}/cosyvoice.yaml via hyperpyyaml with overrides
   {'qwen_pretrain_path': model_dir + '/CosyVoice-BlankEN', 'device': str(device)}; Qwen2Encoder
   does Qwen2ForCausalLM.from_pretrained(pretrain_path).to(device).)
  Lean LLM-only construction (for DPO training, avoids flow/hift/frontend on every rank):
     from hyperpyyaml import load_hyperpyyaml
     with open(COSY_MODEL_DIR + '/cosyvoice.yaml') as f:
         cfg = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': COSY_MODEL_DIR + '/CosyVoice-BlankEN', 'device': str(device)})
     llm = cfg['llm']; llm.load_state_dict(torch.load(COSY_MODEL_DIR + '/llm.pt', map_location=device), strict=False); llm.to(device)
     then load the SFT/DPO ckpt as above. (cfg also instantiates flow/hift on CPU; `del cfg` afterwards.)

Frontend (cosyvoice/cli/frontend.py):
  frontend._extract_text_token(text)  -> (text_token [1,L] int32 on device, text_token_len [1] int32)
      uses self.tokenizer.encode(text) (Qwen tokenizer) — NO normalisation. The package calls this
      directly with text_tts strings and never text_normalize / inference_zero_shot (ttsfrd would
      rewrite and split the SEAME text). The optional --text_frontend flag instead applies
      frontend.text_normalize(text, split=False, text_frontend=True) before tokenising.
  frontend._extract_speech_token(speech16k [1,T])  -> (speech_token [1,N] int32, len)  (audio <= 30 s; 25 tok/s)
  frontend._extract_spk_embedding(speech16k)       -> [1,192]
  frontend._extract_speech_feat(speech24k)         -> (feat [1,F,80], len)
  frontend.frontend_zero_shot(tts_text, prompt_text, prompt_speech_16k, 24000) -> dict with keys
      text, text_len, prompt_text, prompt_text_len, llm_prompt_speech_token, llm_prompt_speech_token_len,
      flow_prompt_speech_token, flow_prompt_speech_token_len, prompt_speech_feat, prompt_speech_feat_len,
      llm_embedding, flow_embedding      (it internally trims prompt tokens/feat so feat = 2*token)
  Audio loading: from cosyvoice.utils.file_utils import load_wav ; load_wav(path, 16000) -> [1,T] float (mono, resampled)

Qwen2LM (cosyvoice/llm/llm.py) facts needed for generation and DPO:
  speech_token_size = 6561 ; EOS id = 6561 ; llm_decoder outputs 6564 classes (ids > 6561 are
  special/fill: generation skips them). llm_embedding.weight[0] = sos/eos emb, [1] = task_id emb.
  Text embedding: llm.llm.model.model.embed_tokens(text_ids)  ; speech embedding: llm.speech_embedding(ids)
  Full-sequence forward: out = llm.llm.model(inputs_embeds=X, attention_mask=M, output_hidden_states=True, return_dict=True)
                         logits = llm.llm_decoder(out.hidden_states[-1])  -> [B,S,6564]
  Incremental: y, cache = llm.llm.forward_one_step(lm_input, masks=tril bool [1,S,S], cache=cache); logits = llm.llm_decoder(y[:, -1])
  Sampler: llm.sampling(weighted_scores=logp[6564], decoded_tokens=list, sampling=25) -> tensor([id])  (RAS top_p=0.8 top_k=25)
           llm.sampling_ids(logp, out_tokens, sampling, ignore_eos=bool) wraps it (ignore_eos re-samples away from EOS)
  Inference conditioning layout (Qwen2LM.inference):
      lm_input = [sos_emb, embed(prompt_text_ids ⊕ text_ids), task_emb, speech_emb(prompt_speech_tokens)]
      then autoregressively generated speech tokens; min_len = 2*len(text_ids), max_len = 20*len(text_ids)
      (ratios relative to the target text only). First token forced non-EOS while i < min_len.
  Training layout (Qwen2LM.forward): lm_input = [sos, embed(text), task, speech_emb(speech)],
      lm_target = [IGNORE]*(1+L) + speech + [6561]; logits[:, i] predicts lm_target[:, i]
      (LabelSmoothingLoss compares x[:, i] with target[:, i]; i.e. logits at the position of the
      task token predict speech[0], logits at speech[t-1] predict speech[t], logits at speech[T-1]
      predict EOS).

Sequence log-prob for DPO (`cosy.sequence_logps`, which follows the inference layout):
  For a candidate c (T tokens) generated with conditioning (text_ids = prompt_text ⊕ text of
  length Lc, prompt_speech tokens of length P):
     X = [sos, embed(text_ids) (Lc), task, speech_emb(prompt_speech (P) ⊕ c (T))]   (length 1+Lc+1+P+T)
     logits = decoder(hidden)  ; start = 1 + Lc + P   (index whose logits predict c[0])
     logp_tok[t] = log_softmax(logits[start+t])[c[t]] for t < T ;  logp_eos = log_softmax(logits[start+T-1+1 -> i.e. start+T])[6561]
     NOTE: logits[start+T] is the position of c[T-1]; its logits predict EOS. So the (T+1)
     scored positions are start .. start+T inclusive. Sequence logp = sum(logp_tok) (+ logp_eos if include_eos).
  Batching right-pads X with zeros and uses attention_mask 1 for valid positions; the per-position
  log-probs are gathered with masks. The optional eos_mask [B] bool overrides include_eos per item
  (False for candidates that never sampled EOS).
  Temperature is NOT applied when scoring (the model's own distribution is scored).

Generation with temperature (`cosy.generate_speech_tokens`): re-implements the loop of
  Qwen2LM.inference (rather than calling it) so that it can (i) divide logits by `temperature` before
  log_softmax, (ii) return the token list, (iii) enforce min/max ratios as in the original,
  (iv) use llm.sampling_ids for RAS sampling with the configured top_p/top_k. Ids > 6561 (fill) are
  MASKED to -inf before sampling (deviation from Qwen2LM.inference's `continue`, which re-feeds a
  stale lm_input and duplicates a KV-cache position). generate_speech_tokens_ex also returns
  ended_with_eos (False = cut by max_len, no EOS sampled).
  The loop runs under torch.inference_mode() and does not go through cv.model.tts (threads + uuid dicts).

Tokens -> waveform (`cosy.tokens_to_wav`) uses cv.model.token2wav:
  key = uuid; cv.model.hift_cache_dict[key] = None
  wav = cv.model.token2wav(token=torch.tensor([tokens], dtype=torch.int32), prompt_token=inputs['flow_prompt_speech_token'],
          prompt_feat=inputs['prompt_speech_feat'], embedding=inputs['flow_embedding'], uuid=key, token_offset=0, finalize=True, speed=1.0)
  cv.model.hift_cache_dict.pop(key); wav: [1, samples] at 24000 Hz on device. (The committed fork's own
  `print('device: ', ...)` in CosyVoice2.__init__ is removed by patches/0001; a working copy of the fork
  may additionally carry an uncommitted debug `print(type(tts_mel))` in token2wav, which is harmless.)
  Resampling to 16 kHz uses torchaudio.functional.resample when needed.

## 4. Whisper-LAL component (code dir $CMI_DPO_LAL_CODE_DIR, default <repo>/lal; env whisperold or the main env)
Reference code (the authors' implementation, vendored verbatim under lal/; only the data-dir defaults of
  train_whisper_LAL.py were changed to read $CMI_DPO_DATA_ROOT): WhisperLAL.py (class
  WhisperWithLAL(nn.Module): .whisper = WhisperForConditionalGeneration, .language_cls = nn.Linear(d_model, 4)),
  LanguageAlignmentLoss, WhisperDataPreLAL.py, train_whisper_LAL.py, utils.py.
Base model $CMI_DPO_LAL_BASE_MODEL, default openai/whisper-small (d_model 768, 80 mel bins, encoder outputs
  1500 frames per 30 s window => 20 ms per frame). It has to be in the HF cache on offline nodes.
Frame language classes (from WhisperDataPreLAL): 0 = zh (Chinese/CJK), 1 = en, 2 = blank/space, 3 = other.
Checkpoint $CMI_DPO_LAL_CKPT is either
  (a) a WHOLE pickled model (torch.save(model)) — the authors' original checkpoints (trained on SEAME, not
      distributed; selected by validation loss / WER over the training run), which unpickle only in a
      torch/transformers close to the one they were saved with (env whisperold), or
  (b) `<ckpt>.state_dict.pt` (+ `<ckpt>.state_dict.json`: base model id, layer_index, n_lang) written once by
      lal/export_state_dict.py in the env that can unpickle (a); loadable in any modern env.
  Loading (lal_cmi.load_lal_model): sys.path.insert(0, LAL_CODE_DIR); import WhisperLAL  # the module name
           stays WhisperLAL for unpickling (a)
           (WhisperLAL.py enables torch.autograd.set_detect_anomaly(True) and CUDA_LAUNCH_BLOCKING at import;
            the loader calls torch.autograd.set_detect_anomaly(False) afterwards and deletes the env var)
           (a) model = torch.load(ckpt, map_location=device); model.eval()
           (b) model = WhisperWithLAL(base ...); model.load_state_dict(torch.load(ckpt)); model.eval()  # base = the
               load_lal_model base_model argument > $CMI_DPO_LAL_BASE_MODEL when set > the file's base_model entry
  Features: 80-mel log spectrogram padded to 30 s, computed with openai-whisper:
           import whisper; audio16k np.float32; feat = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels=80) -> [80,3000]
           (identical to HF WhisperFeatureExtractor output). Batched as [B,80,3000].
  Pseudo labels: enc = model.whisper.model.encoder(feat).last_hidden_state -> [B,1500,768]
                 lang_logits = model.language_cls(enc) -> [B,1500,4]; labels = argmax(-1)
                 n_valid = min(1500, ceil(duration_s / 0.02)); keep labels[:n_valid].
CMIspeech(u) = (T(u) - max_k T_k(u)) / T(u), k in L. Default L = {zh(0), en(1)}: frames labelled
  blank(2)/other(3) are excluded from T(u) (flag --count_all_classes includes all 4). If T(u)==0 -> CMI 0.
  ΔCMI = |CMIspeech(synth) - CMIspeech(gt)|. CMI values are reported in [0,1]; printed/aggregated as %.
A token-level text CMI (classic, from text_tts: CJK-char tokens = zh, ASCII-letter tokens = en) is
  provided as a sanity column (common.text_cmi).
Output TSV (13_score_cmi.py): columns  utt  cand  wav  n_frames  n_zh  n_en  n_blank  n_other  cmi
  where cand = 'gt' for ground truth rows or the candidate index; plus an optional RLE column of
  labels when --dump_labels.

## 5. UTMOS critic (env cosyvoicenew; pure torch+torchaudio)
  Two sources, selected by --utmos_source / $CMI_DPO_UTMOS_SOURCE (hub | local, default hub):
  hub:   m = torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True)  (internet once;
         the torch.hub cache is reused afterwards)
  local: UTMOS_REPO = $CMI_DPO_UTMOS_REPO   (a local clone of SpeechMOS; sys.path.insert(0, UTMOS_REPO))
         from speechmos_ut.utmos22.strong.model import UTMOS22Strong
         UTMOS_CKPT = $CMI_DPO_UTMOS_CKPT   (the utmos22_strong state_dict torch.hub would download)
         m = UTMOS22Strong(); m.load_state_dict(torch.load(UTMOS_CKPT, map_location='cpu')); m.eval().to(device)
  Both build the same class with the same weights (the authors' offline nodes used local).
  score = m(wave [B,T] float32, sr)  -> [B] MOS (1..5). It resamples internally to 16 kHz. Utterances are
  scored one at a time (variable length), inside torch.no_grad().
  Output TSV: utt  cand  wav  utmos

## 6. MER critic (env asr-whisper)
  ASR_MODEL_DIR = $CMI_DPO_ASR_MODEL_DIR   (HF dir: whisper-large-v3 fine-tuned on SEAME, contains
  config/tokenizer/preprocessor/generation_config/model.safetensors; trained on SEAME, not distributed)
  Decoding is identical to the authors' SEAME Whisper baseline decoder:
     from transformers import pipeline
     asr = pipeline('automatic-speech-recognition', model=ASR_MODEL_DIR, torch_dtype=torch.float16, device=0, chunk_length_s=30, batch_size=B)
     outputs = asr(generator of {'array': np.float32 16k, 'sampling_rate': 16000}, batch_size=B, generate_kwargs={'task': 'transcribe'})
     text = out['text'].strip()
  SEAME MER normaliser (copy of the authors' SEAME scorer, make_seame_normalizer):
     _CJK = re.compile(r"([㐀-䶿一-鿿豈-﫿])"); _TAG = re.compile(r"<[^>]*>"); _APOS = re.compile(r"['’]")
     s.lower(); tags->' '; opencc t2s if available; apostrophes removed; each CJK char isolated with spaces;
     re.sub(r"[^\w ]", " ", s, flags=re.UNICODE); '_'->' '; collapse spaces.
  MER(utt) = levenshtein(ref_tokens, hyp_tokens) / len(ref_tokens)   (word-level over the normalised
  token strings; can exceed 1.0). Corpus MER = sum(edits)/sum(ref_len). levenshtein lives in
  common.py (no jiwer dependency); 11_score_mer.py --jiwer_check additionally cross-checks the corpus
  value with jiwer when it is importable.
  Output TSV: utt  cand  wav  hyp  ref  n_ref  edits  mer

## 7. Preference pairs (14_build_pairs.py) — paper §3.2.3
  Inputs: cands.tsv (utt cand wav dur n_tokens), tokens/<utt>.pt (needed for the pool filter and the
  tensors), mer.tsv, utmos.tsv, cmi.tsv (cands + gt rows). Flags --cands --tokens_dir --out are required;
  --mer / --utmos / --cmi are required ONLY while the matching weight (--gam / --lam / --nu) is non-zero.
  0. Ranking pool per utt: a candidate is EXCLUDED (neither pos nor neg, and left out of the normalisation
     ranges) when n_tokens > --max_cand_tokens (default 750 = 30 s at 25 tok/s) or when its ended_with_eos
     flag in tokens/<utt>.pt is False (cut at max_len; --keep_truncated keeps those). Counted in
     candidate_counts.cand_pooled / cand_excluded_over_max_tokens / cand_excluded_truncated / cand_truncated_kept.
  1. ΔCMI(utt,cand) = |cmi(utt,cand) - cmi(utt,'gt')|  (only when dCMI is active).
  2. MER clip: mer_clipped = min(mer, --mer_clip) (default 1.0; `inf` disables) BEFORE normalisation
     (raw MER is unbounded; one hallucinated candidate at 10-20 would squeeze every other candidate to
     mer_n ~ 0). --max_mer and the mer_pos / mer_neg values in pairs.pt use the RAW MER. Summary key
     mer_clip.n_clipped.
  3. Normalise each active critic to [0,1]: --norm global (min-max over all pooled candidates; default)
     or per_utt; a zero range maps to 0.
  4. R = λ·S~utmos − γ·S~mer − ν·S~Δcmi   (defaults λ=γ=ν=1.0; flags --lam --gam --nu).
  DISABLED CRITIC (weight 0) semantics (paper Table-1 rows MER only / MER+UTMOS / all three): the critic
     contributes exactly 0 to R; its TSV is optional and missing rows drop nothing (value recorded as
     null / None); its threshold is NOT applied unless --threshold_disabled_critics (then a preferred
     candidate without a value is dropped as pos_critic_missing); with --nu 0 the gt-CMI merge is skipped
     entirely (cmi.tsv may lack gt rows, no missing_gt_cmi drops). With all weights > 0 every critic TSV is
     required and a candidate is scored only when all three critics and the utt's gt CMI are present.
     The active critics are logged and stored in pairs_summary.json (active_critics).
  5. Per utt (>= 2 pooled candidates): pos = argmax R, neg = argmin R (ties towards the lowest index).
  6. Drop the pair if R_pos == R_neg (tie) or the PREFERRED candidate violates an ACTIVE critic's threshold:
     raw mer > 0.20, utmos < 2.5, Δcmi > 0.20 (flags --max_mer --min_utmos --max_dcmi; inf relaxes one).
  Utterance drop reasons, in test order: missing_tokens (the tokens file is loaded for EVERY utt of cands.tsv
  before the critic join, so an utt without critic rows and without a tokens file is missing_tokens),
  missing_gt_cmi (dCMI active only), tokens_index_missing (>= 2 critic-complete candidates but < 2 of them
  indexable in tokens/<utt>.pt), too_few_candidates (< 2 scored = critic-complete AND indexable, before the
  pool filter), too_few_after_pool_filter, tie, pos_mer_above_max, pos_utmos_below_min, pos_dcmi_above_max,
  pos_critic_missing, tokens_index_missing again (selected index gone at the final tokens load).
  Output pairs.pt: list of dicts {utt, text_ids (LongTensor, prompt_text⊕text as used at generation),
    prompt_speech (LongTensor), pos (LongTensor), neg (LongTensor), eos_pos, eos_neg (bool; True when the
    tokens file lacks ended_with_eos), r_pos, r_neg, mer_pos, utmos_pos, dcmi_pos, cand_pos, cand_neg,
    mer_neg, utmos_neg, dcmi_neg}; the mer_/utmos_/dcmi_ fields of a disabled critic may be None
    (15_train_dpo.py never reads them).
  pairs_summary.json: config (incl. mer_clip, threshold_disabled_critics, max_cand_tokens, keep_truncated),
    active_critics, pool_filter, mer_clip, n_pairs, dropped_utts per reason, candidate_counts (the Counter is
    pre-seeded, so these keys are ALWAYS present, 0 when nothing was counted: cand_rows, cand_scored,
    cand_pooled, cand_excluded_over_max_tokens, cand_excluded_truncated, cand_truncated_kept,
    cand_no_tokens_file, cand_index_not_in_tokens, cand_missing_mer / cand_missing_utmos / cand_missing_cmi +
    cand_missing_gt_cmi for active critics, cand_missing_<critic>_ignored for disabled ones; other keys such as
    cand_duplicate_rows, n_tokens_mismatch, eos_flag_missing, truncated_pos/neg appear only when non-zero),
    normalisation ranges, mean critic values for pos/neg. Fully deterministic (no RNG).
  Slurm: pairs.sbatch merges CMI_GT into <out>/cmi_merged.tsv only while NU != 0; a weight-0 critic's TSV is
    passed only when the file exists (informational).

## 8. Candidate generation (10_gen_candidates.py) — paper §3.2.1
  For each manifest row (utt, text_tts, prompt_utt/prompt_wav/prompt_text): build inputs via the
  frontend (prompt speech tokens/feat/embedding once per utt), generate N candidates with
  temperature τ (defaults N=4, τ=1.0; flags --n_cand --temperature --top_p --top_k --sampling --seed),
  vocode each with tokens_to_wav, save:
     <out>/wav/<utt>/<k>.wav  (24 kHz, k = 0..N-1)      (--save_16k also writes <k>_16k.wav)
     <out>/tokens/<utt>.pt  = {'utt','text_ids' (prompt⊕text LongTensor), 'text_len_target', 'prompt_speech' (LongTensor), 'cands': [LongTensor]*N, 'ended_with_eos': [bool]*N, 'prompt_utt'}
     <out>/cands.tsv        rows: utt  cand  wav  dur  n_tokens   (append-safe; skip utts already done => resumable)
     <out>/failed.txt       utt <TAB> reason
     <out>/gen_config.json  CONFIG FINGERPRINT: ckpt (path, mtime, size), temperature, n_cand, sampling,
                            top_p/top_k (null = model default), seed, text_frontend, text_mode (inferred
                            from the manifest), manifest_sha256 (common.manifest_content_hash: sha256 over
                            the sorted utt/text_raw/text_tts/prompt_utt/prompt_wav/prompt_text rows, i.e.
                            the manifest CONTENT that conditions generation, order-independent; the manifest
                            path and row count are stored outside 'config' for information only). On a
                            resume the current config is compared with the stored one and the run ABORTS
                            (rc 1) on any difference unless --force_resume (then the differences are logged
                            and the fingerprint overwritten). A stored fingerprint that lacks a newer key
                            (older run) only warns for that key and is re-written. Outputs that predate the
                            fingerprint only warn. All shards write the same fingerprint.
  Text: text_tts and prompt_text are taken from the manifest VERBATIM (00 already applied --text_mode to
  both columns; nothing is re-stripped here).
  --ckpt <path> (default SFT_LLM_CKPT); --ckpt none = the stock CosyVoice2 llm.pt (the flag has to be passed
  explicitly: omitting it selects the SFT model). Candidate k of utt u re-seeds all RNGs from
  sha256(seed, u, k, attempt); --max_retries (2) fresh-seed retries on a sampler RuntimeError.
  Sharding: --shard i --nshards n splits the manifest across jobs (<= 6 GPUs); each shard appends to its
  own cands.tsv.<shard> / failed.txt.<shard>; `--merge` (no GPU) concatenates, de-duplicates on (utt, cand)
  and drops rows without tokens/<utt>.pt into the final cands.tsv. --limit for smoke tests.
  --dry_run: fingerprint + plan (rows with done/pending status) without loading the model, rc 0 (a
  fingerprint mismatch still aborts with rc 1). Exit 1 when every attempted utterance failed.
  Log ends with `DONE shard=i/n rows= done= skipped= failed= cands_written= rows_already_present= out=`.

## 9. Stage-1 SFT (01_build_sft_cache.py + 02_train_sft.py) — paper §4
  Cache: for each manifest row: text_token via frontend._extract_text_token(text_tts), speech_token via
  frontend._extract_speech_token(load_wav(wav,16000)); skip if audio > 30 s or text tokens > 200 or < 1.
  Saves a list of {'utt','text_token'[L] int32 cpu,'speech_token'[T] int32 cpu} to <out>.pt (sharded ok).
  Train: torchrun DDP over the LLM only (Qwen2LM.forward(batch, device) returns {'loss','acc'}; batch keys
  text_token [B,L], text_token_len [B], speech_token [B,T], speech_token_len [B]; padded with 0),
  init from COSY_MODEL_DIR/llm.pt (already loaded by construction; --init_ckpt overrides), AdamW peak lr
  2e-4 (paper). Schedule --lr_schedule: constant_with_warmup (DEFAULT: linear warm-up over --warmup_steps
  then constant; the paper only states a warm-up) | linear (warm-up then decay to 0 = the previous
  hard-coded behaviour); --warmup_steps -1 = 10 % of the total optimizer steps.
  Batch: --batch is the micro-batch PER GPU (paper "batch size of 4" reproduced per GPU); effective batch
  = batch × world_size × --accum (default accum 1). --global_batch G fixes the effective batch instead:
  accum = G / (batch × world_size), which has to divide exactly (parse error otherwise), overrides --accum.
  --max_steps (~50k) / --eval_every / --log_every / --warmup_steps count OPTIMIZER steps. Eval on the valid
  cache each --eval_every, early stopping on val loss (--patience, --min_delta), save state_dict per
  epoch/best as <exp>/epoch_XXX.pth and <exp>/best.pth (keys WITHOUT 'module.' prefix). Next to every
  checkpoint rank 0 atomically (over)writes <exp>/train_state.pt (~4 GB for the 505M model:
  {ckpt, optimizer, scheduler, epoch, micro, step, best_val, best_step, bad_evals, epoch_end_done,
  geometry = {world_size, batch, accum (resolved), seed}}; epoch_end_done False for a best.pth written by a
  validation, True for an epoch-end save).
  --resume <ckpt>: loads its weights (instead of --init_ckpt) and, when train_state.pt names that
  checkpoint, optimizer / scheduler / counters / best val, then skips the consumed epochs and micro-batches
  (deterministic DistributedSampler order given --seed); train_state.pt missing = weights-only resume with a
  warning; naming ANOTHER checkpoint = error; parse_args errors when the file does not exist. Geometry check:
  a mid-epoch state (micro > 0) whose geometry differs from the current world_size / --batch / accum / --seed
  is refused (SystemExit: the replay would skip the wrong micro-batches); at an epoch boundary (micro 0) a
  change only warns; a state without geometry (older run) only warns. A best.pth written on the last
  optimizer step of an epoch (micro == n_micro, epoch_end_done False) resumes into that epoch's pending
  epoch-end block (epoch_XXX.pth + boundary sidecar written; the validation is not repeated), also when
  that step was the last one allowed by --max_steps / --epochs, instead of skipping it.
  Device: CUDA/NCCL when available, else CPU/gloo (smoke only). The training loop mirrors the structure
  of the CosyVoice fork's train.py.

## 10. DPO training (15_train_dpo.py) — paper §2 eq.(1)
  policy = LLM initialised from --init_ckpt (default SFT_LLM_CKPT; 'none' (any case) or '' = the STOCK
  CosyVoice2 llm.pt via cosy.load_llm_only(device, llm_ckpt=None)); ref = frozen deep copy of the policy
  right after that load (eval, no grad), i.e. always the --init_ckpt model.
  loss = -logσ( β[(logπ(pos)-logπref(pos)) - (logπ(neg)-logπref(neg))] ), β default 0.1 (flag).
  Optional --sft_weight w adds w * (-logπ(pos)/len) regulariser (default 0). Optional --length_norm.
  Batch of pairs (default 4 per GPU), --accum, AdamW lr 1e-6 constant (default), grad clip 1.0, epochs
  (default 2), DDP via torchrun like train.py (policy wrapped in DDP; ref not). Device: cuda:<LOCAL_RANK> /
  NCCL when CUDA is available, else CPU/gloo automatically (no flag).
  Precision: --bf16 applies to the POLICY forward only (training + validation). The REFERENCE log-probs
  (the --cache_ref table, default on, and any on-the-fly forward with --no-cache_ref) are ALWAYS float32
  with autocast disabled. Step-0 margin (policy == reference weights): exactly 0 (loss log 2) ONLY when the
  policy also runs in float32 (no --bf16) AND the training micro-batch has the cache's padding layout
  (--batch 1 or identical batch composition; float32 kernels differ by ~1e-5 across paddings). Under the
  production setting (--bf16, dpo.sbatch BF16=1) the policy side is bf16, so the step-0 margin is bf16
  rounding noise (loss ~0.69 +- small), not 0. A fresh run with --cache_ref logs on rank 0
  `step-0 check ... max|logp_policy - logp_ref_cache| = ...` (first training micro-batch through the policy
  under the training autocast) so the real discrepancy is visible.
  Pair filtering: --max_pair_tokens (default 750; <= 0 off) skips at load time every pair whose pos or neg
  has more speech tokens (logged count); pairs with an empty pos/neg are skipped too.
  Model selection: --val_frac (default 0.05) holds out a seeded fraction of the loaded pairs that is never
  trained on: n_val = max(1, round(val_frac * n)) when val_frac > 0 and n_pairs >= 20, else 0. Rank 0 writes
  the held-out utts to <exp>/val_pairs.txt (always, may be empty). After every epoch the held-out DPO loss /
  margin / accuracy are computed (sharded, all-reduced) and appended to metrics.jsonl (val_loss, val_margin,
  val_acc, n_val, plus best_loss and selection); dpo_best.pth = lowest VALIDATION loss when a validation set
  exists, else the lowest epoch-mean training loss.
  Log: loss, reward margin, reward accuracy (fraction margin > 0), every --log_every optimizer steps.
  Save <exp>/dpo_epoch_XXX.pth, <exp>/dpo_best.pth, <exp>/dpo_step_XXXXXX.pth (--save_every) (state_dict
  without 'module.') loadable by the same loader as the SFT ckpt (§3). Next to every
  dpo_epoch/dpo_step checkpoint: <exp>/train_state.pt {ckpt, optimizer, epoch, micro, global_step,
  best_loss (= the selection metric), epoch_end_done (False for a dpo_step save, True for an epoch-end
  save), geometry = {world_size, batch, accum, seed, val_frac, max_pair_tokens, n_pairs}} and, once,
  <exp>/ref_cache.pt (float32 reference log-probs of ALL loaded pairs, with init_ckpt, max_pair_tokens,
  dtype and pair-count keys; an old ref_cache.pt without max_pair_tokens is still accepted).
  --resume <ckpt in exp>: weights into the policy only, optimizer / counters from train_state.pt when it
  names that checkpoint, ref_cache.pt reused when it matches, consumed epochs/micro-batches skipped,
  metrics.jsonl appended (truncated otherwise). Geometry check: the state is REFUSED (SystemExit) when
  seed / val_frac / max_pair_tokens / n_pairs differ (the held-out split would change and held-out pairs
  could be trained on) and when world_size / batch / accum differ while micro > 0 (the micro-batch replay
  is partition-specific); another GPU count / batch is accepted with a warning from an epoch-boundary state
  (micro 0, dpo_epoch_XXX.pth of a completed epoch); a state without geometry (older run) only warns.
  Pending epoch end: a dpo_step save on the last optimizer step of an epoch has micro == n_micro and
  epoch_end_done False; resuming from it runs that epoch's validation, writes dpo_epoch_XXX.pth (+ dpo_best
  on a validation-loss improvement; without a validation set the epoch's training loss is unknown, so
  dpo_best is left unchanged and metrics.jsonl records selection 'none', epoch_loss null) and continues.

## 11. Synthesis for augmentation (20_synthesize.py)
  manifest (train transcripts) + prompt assignment + --ckpt (SFT or DPO; 'none' = stock llm.pt) ->
  synthetic corpus: <out>/wav/<utt>.wav (16 kHz, --sr flag), <out>/wav.scp, <out>/text (text_raw),
  <out>/utt2spk, <out>/utt2dur, <out>/failed.txt, <out>/synth_config.json. One candidate per transcript.
  text_tts / prompt_text are taken from the manifest verbatim (no re-stripping).
  Row order: the manifest rows are SHUFFLED with random.Random(--seed) BEFORE sharding, --limit and the
  --hours cap (speaker-balanced shards and capped subsets; --no_shuffle keeps manifest order); the merge /
  finalize pass sorts the Kaldi files by utt. --seed therefore also seeds the shuffle.
  Quality filter (Whisper fine-tuning takes <= 30 s clips): a draw is REJECTED when generation hit max_len
  without EOS (ended_with_eos False; --keep_truncated disables) or when the vocoded wav is longer than
  --max_dur s (default 30.0; `inf` disables; has to be > 0). A rejected draw is re-sampled with a fresh seed
  --reject_retries times (default 1); if every draw is rejected the utt goes to failed.txt as
  "<utt>\t rejected: <reason>" and no wav / Kaldi line is written. On a resume an utt whose LAST line in
  this shard's failed file is a "rejected:" line counts as done (deterministic draws; the DONE line reports
  rejected_skipped=); --retry_rejected re-draws them (only useful with a changed --reject_retries /
  --keep_truncated / --max_dur, i.e. with --force_resume). Other failures are retried on every resume.
  --hours cap: corpus-level; each shard stops at hours / nshards (durations already in the shard's utt2dur
  count on resume). --shard/--nshards, --temperature, --sampling, --top_p/--top_k, --max_retries (2),
  --utt_prefix (e.g. syn_), resumable (an utt counts as done only when present in all four shard files with
  its wav on disk). Fingerprint <out>/synth_config.json (ckpt path/mtime/size, temperature, sampling,
  top_p/top_k, seed, text_frontend, text_mode, max_dur, keep_truncated, shuffle, sr, utt_prefix,
  manifest_sha256 = common.manifest_content_hash of the manifest content, as in §8): a resume with a
  different config ABORTS unless --force_resume (fingerprint overwritten); a stored fingerprint lacking a
  newer key only warns and is re-written. --dry_run prints the fingerprint + plan (done / rejected /
  pending / beyond_cap) without loading the model, rc 0. --merge also merges <out>/failed.txt.<shard> (+ an
  existing <out>/failed.txt) into <out>/failed.txt: one line per utt (last wins), utts present in the merged
  corpus removed; the --nshards 1 finalize pass de-duplicates the plain failed.txt likewise.
  Log ends with `DONE shard=i/n rows= done= skipped= failed= rejected= rejected_skipped= redraws=
  hours_in_shard= capped= out=` (merge: `DONE merge out= utts= dropped_incomplete= failed_utts=`). Exit 1
  when every attempted utt failed or was rejected.

## 12. Evaluation of a TTS system (eval_tts.py) — reproduces paper Table 1
  Given a synthesized set dir (with cands.tsv or wav.scp) and the three critic TSVs, print
  mean UTMOS, corpus MER (%), mean ΔCMI (%) and write eval.json. Works on devman/devsge manifests.

## 13. Manifest (00_prep_manifest.py)
  Input: Kaldi dir. Output TSV (tab-separated, header):
     utt  spk  dur  wav  text_raw  text_tts  text_ref  prompt_utt  prompt_wav  prompt_text
  plus <out stem>.json ({n_utts, hours, n_speakers, n_dropped_no_prompt, n_available, hours_available,
  hours_cap_selects_all, n_selected, hours_selected, prompt_counts, mean_text_cmi_pct, args}) and
  <out stem>.dropped.txt ("utt<TAB>reason" per dropped target; always written, may be empty);
  <out stem> = --out without its extension (m.tsv -> m.json / m.dropped.txt).
  Prompt assignment (--prompt_mode same_spk (default) | self): same speaker, different utt, duration
  in [--prompt_min 3, --prompt_max 10] s, chosen with a seeded RNG; fallback to the other utt of the
  speaker whose duration is closest to the window (prompt pool restricted to [--min_dur, 30] s: the
  CosyVoice frontend asserts prompt audio <= 30 s; train has 21 utts > 30 s and utts down to 0.08 s).
  A target whose speaker has NO other pool utterance is DROPPED by default (reason no_other_prompt_utt,
  counted in the log / json); --allow_self_prompt falls back to the utterance itself instead (prompt ==
  target, leaks the target into the prompt; the earlier silent behaviour). --prompt_mode self makes every utt
  its own prompt (nothing dropped). --prompt_pool all (default): prompts are drawn from the WHOLE Kaldi dir,
  not only from the --hours/--limit subset; subset: pool = the selected targets (self-contained manifest).
  --min_dur/--max_dur filter targets (default 1 .. 30 s); parse_args rejects prompt_min > prompt_max,
  min_dur > max_dur, prompt_max > 30 and --hours <= 0. --text_mode strip|raw (§2), applied to text_tts AND
  prompt_text. --hours cap and --seed for subset selection: SEAME train filtered to [1, 30] s is ~94.3 h, so
  --hours 100 selects everything (the log and json say so: hours_cap_selects_all); the paper's "100 h real"
  is therefore the whole filtered train. Prompt drops happen AFTER --hours/--limit, so the written hours can
  be slightly below the cap. --limit for smoke tests.

## 14. Package layout (repository root = <repo>, the directory holding docs/)
  docs/DESIGN.md (this file)   README.md (public: install, configuration, inference)   docs/PIPELINE.md (training,
  synthesis, evaluation recipes)   docs/CLUSTER.md (SLURM layer, legacy envs)
  cmi_dpo/__init__.py
  cmi_dpo/paths.py           configuration: REPO_ROOT, ENV_PREFIX='CMI_DPO_', load_env_file() (reads
                             <repo>/config/paths.env at import unless CMI_DPO_NO_ENV_FILE=1; os.environ wins over the
                             file), get(name, default), require(name) (SystemExit naming CMI_DPO_<name>,
                             config/paths.env and config/paths.env.example), describe(), and the accessors
                             cosy_root() cosy_model_dir() sft_ckpt() lal_code_dir() lal_ckpt() lal_base_model()
                             asr_model_dir() utmos_source() utmos_repo() utmos_ckpt() data_root()
  cmi_dpo/common.py          manifest IO, kaldi readers, text_tts/text_ref, seame_normalize, levenshtein, tsv helpers, seed_all, resample
  cmi_dpo/cosy.py            load_cosyvoice2, load_llm_only, load_llm_ckpt, build_prompt_inputs, generate_speech_tokens, tokens_to_wav, sequence_logps
  cmi_dpo/lal_cmi.py         load_lal_model (pickle or state_dict), whisper_mel, frame_language_labels, cmi_from_labels, text_cmi, delta_cmi
  lal/                       the authors' Whisper-LAL code (WhisperLAL.py WhisperDataPreLAL.py train_whisper_LAL.py utils.py; verbatim
                             except train_whisper_LAL.py's data-dir defaults = $CMI_DPO_DATA_ROOT/{train,valid,devman,devsge})
                             + export_state_dict.py + README.md (training / export)
  scripts/00_prep_manifest.py  01_build_sft_cache.py  02_train_sft.py  10_gen_candidates.py  11_score_mer.py
          12_score_utmos.py  13_score_cmi.py  14_build_pairs.py  15_train_dpo.py  20_synthesize.py  eval_tts.py
          infer_one.py (single-sentence inference: --text --prompt_wav --prompt_text [--ckpt] --out ...)
  config/paths.env.example   every CMI_DPO_* variable with a comment; copied to the git-ignored config/paths.env
  env/environment.yml env/requirements.txt env/freeze/{cosyvoicenew,asr-whisper,whisperold}.txt
  patches/0001-cosyvoice-fork-remove-debug-print.patch   applied to the CosyVoice fork clone
  slurm/load_env.sh (sourced loader of <repo>/config/paths.env that keeps CMI_DPO_* variables already set: env wins, as in paths.py)
  slurm/sb (sbatch wrapper adding the site options from config/paths.env; usage `slurm/sb <file.sbatch> [sbatch args]`)
        prep.sbatch sft_cache.sbatch sft.sbatch gen.sbatch mer.sbatch utmos.sbatch cmi.sbatch pairs.sbatch dpo.sbatch synth.sbatch eval.sbatch
        (headers hold only job-name/gres/time/cpus; bodies start with PKG=${CMI_DPO_PKG:?run through slurm/sb};
        source "$PKG/slurm/load_env.sh"; source "$CMI_DPO_CONDA_SH"; conda activate "$CMI_DPO_ENV_MAIN|ASR|LAL")
        run_dpo_round.sh (submits gen -> {mer,utmos,cmi} -> pairs -> dpo with --dependency through slurm/sb; pins every
        variable of every stage in its --export list, `none` = empty; the per-stage export strings are
        expanded QUOTED so a multi-word *_EXTRA stays one --export argv word; every sbatch maps `none` to
        "" for its optional variables, mer.sbatch's MANIFEST=none only with WAV_SCP)  smoke.sh (4-utt end-to-end,
        srun options from the same variables)
  logs/ data/ exp/ smoke_out/ third_party/ pretrained_models/   git-ignored run-time directories
  Scripts import the package via: sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  then `from cmi_dpo import common, paths` etc. Every script has argparse with the defaults above (path
  defaults from paths.py) and a module docstring stating the env it runs in.

## 14a. Release layout (public GitHub copy, repo name CSTTS-DPO)
  Tracked: the files listed in §14 except the git-ignored directories, config/paths.env, *.pt/*.pth/*.wav/*.flac.
  Server-agnostic by construction: every machine-specific location is a CMI_DPO_* variable, and no tracked
  file contains a machine-specific path prefix. The CosyVoice fork is not vendored: it is cloned from
  https://github.com/YUCHEN005/TTS_finetune into third_party/TTS_finetune, patches/0001 is applied (repo-root paths,
  i.e. `git -C third_party/TTS_finetune apply ../../patches/0001-...patch`) and Matcha-TTS is cloned into
  third_party/TTS_finetune/CosyVoice/Matcha-TTS pinned to the upstream commit that set matcha/VERSION to 0.0.5.1 (the
  fork does not track it and the authors' copy carries no git metadata); CMI_DPO_COSY_ROOT is then set to the ABSOLUTE
  path <repo>/third_party/TTS_finetune/CosyVoice (paths.py does not resolve relative values). Pretrained models are
  downloaded at installation time (CosyVoice2-0.5B via modelscope, whisper-large-v3 / whisper-small via HF, UTMOS via
  torch.hub); the SEAME-fine-tuned checkpoints (stage-1 LLM, Whisper-LAL, MER critic) are NOT distributed (SEAME licence);
  the LLM and Whisper-LAL are rebuilt with docs/PIPELINE.md / lal/README.md, the MER critic is any SEAME-fine-tuned HF
  Whisper dir (whisper-large-v3 + transformers Trainer for the authors; no recipe in this repo). Licence Apache-2.0
  (LICENSE, NOTICE), CITATION.cff.
  The cluster copy keeps working unchanged with config/paths.env holding the server values; slurm/smoke.sh
  exercises the whole chain end to end on a 4-utterance fixture.

## 15. Function reference (library API, exact signatures)
common.py
  read_kaldi_dir(d: str) -> dict[utt] = {'wav','text_raw','spk','dur'}
  strip_tags(text: str) -> str                      # text_tts
  seame_normalize(text: str) -> str                 # text_ref / hyp normalisation (§6), opencc optional
  opencc_available() -> bool
  levenshtein(a: list[str], b: list[str]) -> int
  mer_pair(ref: str, hyp: str) -> tuple[int edits, int n_ref, float mer]    # on normalised strings
  read_tsv(path, strict: bool = False) -> list[dict]; write_tsv(path, rows: list[dict], columns: list[str]); append_tsv(path, row, columns)
  repair_torn_tsv(path: str, n_columns: int) -> int # truncate a last line torn by a hard kill; bytes removed (critics' --resume)
  append_failed(path, utt, cand, wav, reason) -> None; reset_failed(path) -> None   # <out>.failed.txt of the critics
  failure_exit_code(n_failed: int, n_total: int, max_fail_frac: float, strict: bool, failed_txt: str, log=None) -> int
      # shared critic policy: 0 while n_failed / max(n_total, 1) <= max_fail_frac (WARNING), else 2; strict -> 2 on any failure
  read_manifest(path) -> list[dict]; write_manifest(path, rows)
  manifest_content_hash(rows) -> str   # sha256 over sorted MANIFEST_HASH_COLUMNS rows (utt text_raw text_tts
                                       # prompt_utt prompt_wav prompt_text); order-independent; §8/§11 manifest_sha256
  load_audio16k(path) -> np.ndarray float32 (mono, 16 kHz)   # soundfile/torchaudio; used by critics
  resample(wav: torch.Tensor [1,T], orig: int, target: int) -> torch.Tensor
  seed_all(seed: int)
  shard(rows: list, i: int, n: int) -> list         # round-robin
  token_langs(text_tts: str) -> list[int]; text_cmi(text_tts: str) -> float   # token-level CMI (CJK char = zh, ASCII-letter word = en)
  hours(rows) -> float; setup_logging(level)
cosy.py  (imports cosyvoice lazily inside functions; adds COSY_ROOT paths)
  COSY_ROOT, COSY_MODEL_DIR, SFT_LLM_CKPT constants (resolved from paths.py: COSY_ROOT / COSY_MODEL_DIR are
  required at use time, SFT_LLM_CKPT is None when CMI_DPO_SFT_CKPT is empty)
  load_cosyvoice2(device, llm_ckpt: str | None = SFT_LLM_CKPT) -> CosyVoice2
  load_llm_only(device, llm_ckpt: str | None) -> Qwen2LM        # hyperpyyaml lean path; None = stock llm.pt only
  load_llm_ckpt(llm, ckpt: str) -> None                          # strips 'module.', strict=True
  configure_sampling(llm, top_p: float = 0.8, top_k: int = 25, win_size: int = 10, tau_r: float = 0.1) -> None   # --top_p/--top_k overrides
  build_prompt_inputs(cv, text_tts: str, prompt_text_tts: str, prompt_wav: str, text_frontend: bool = False) -> dict
      returns the frontend_zero_shot dict (built manually via _extract_* so that no ttsfrd normalisation
      happens unless text_frontend=True) PLUS 'text_ids' (LongTensor [Lc] = prompt_text ⊕ text ids, cpu),
      'text_len_target' (int L of the target text), 'prompt_speech' (LongTensor [P], cpu)
  generate_speech_tokens(llm, inputs: dict, temperature: float = 1.0, sampling: int = 25,
      min_token_text_ratio: float = 2, max_token_text_ratio: float = 20) -> list[int]
  generate_speech_tokens_ex(same args) -> tuple[list[int], bool ended_with_eos]   # False = cut by max_len (10 stores it, 14 pool-filters on it, 20 rejects on it)
  tokens_to_wav(cv, tokens: list[int], inputs: dict) -> torch.Tensor [1,S] float32 cpu @ 24 kHz
  sequence_logps(llm, text_ids: LongTensor [B,Lc] (right-padded), text_lens: LongTensor [B],
      prompt_speech: LongTensor [B,P] (right-padded), prompt_lens: LongTensor [B],
      cand: LongTensor [B,T] (right-padded), cand_lens: LongTensor [B], include_eos: bool = True,
      return_lengths: bool = False, eos_mask: BoolTensor [B] | None = None) -> Tensor [B] (sum log-prob)
      Because prompt lengths vary, each sequence's embeddings are built separately (list), then padded to a
      batch (like Qwen2LM.pad_unpad_sequence) and the per-position gather is computed with masks. Called under
      autocast(bf16) for the policy only when --bf16; always float32 for the reference (§10).
lal_cmi.py (env whisperold for pickled checkpoints; any env for exported state dicts)
  LAL_CKPT_DEFAULT constant (= paths.lal_ckpt() when set); ZH, EN, BLANK, OTHER = 0,1,2,3
  load_lal_model(ckpt: str, device) -> torch.nn.Module (eval)   # pickle, or <ckpt>.state_dict.pt (+ .json) from lal/export_state_dict.py
  whisper_mel(audio16k: np.ndarray) -> torch.Tensor [80,3000]
  frame_language_labels(model, mel_batch: Tensor [B,80,3000], durations_s: list[float]) -> list[np.ndarray int]
  cmi_from_labels(labels: np.ndarray, langs=(0,1)) -> tuple[float cmi, dict counts]
  delta_cmi(cmi_synth: float, cmi_gt: float) -> float
  labels_rle(labels) -> str; rle_to_labels(rle: str) -> np.ndarray
