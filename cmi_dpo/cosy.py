"""CosyVoice2 access layer for the cmi_dpo package (see docs/DESIGN.md).

Purpose
-------
Thin wrappers around a CosyVoice2 checkout (``CMI_DPO_COSY_ROOT``, see ``cmi_dpo/paths.py``) that
the candidate-generation, DPO-training and synthesis scripts code against:

  load_cosyvoice2        full CosyVoice2 (frontend + LLM + flow + hift) with the SFT LLM loaded
  load_llm_only          lean Qwen2LM-only construction via hyperpyyaml (DPO ranks)
  load_llm_ckpt          load a DDP / plain state_dict into a Qwen2LM (strips 'module.', strict)
  configure_sampling     rebind the RAS sampler with explicit top_p / top_k
  build_prompt_inputs    frontend_zero_shot's tensor dict built manually (no ttsfrd rewrite)
                         plus 'text_ids' / 'text_len_target' / 'prompt_speech' cpu LongTensors
  generate_speech_tokens Qwen2LM.inference's loop re-implemented with a temperature knob
  generate_speech_tokens_ex  same, also returning whether EOS terminated the sequence
  tokens_to_wav          speech tokens -> 24 kHz waveform through cv.model.token2wav
  sequence_logps         batched, differentiable sequence log-probabilities (DPO policy / ref)

Environment: the CosyVoice env (torch 2.3.1, transformers 4.40.1, hyperpyyaml). All cosyvoice
imports are lazy (inside functions) and go through ``_ensure_cosy_path``. Locations come from
``cmi_dpo.paths`` and are resolved lazily: importing this module never needs the CMI_DPO_*
environment variables; ``COSY_ROOT`` / ``COSY_MODEL_DIR`` / ``SFT_LLM_CKPT`` stay available as
module attributes (evaluated on access, see ``__getattr__``).

Caveat: loading ``cosyvoice.yaml`` (both loaders below) executes the yaml's
``!apply:random.seed [1986]`` / numpy / torch seed lines, i.e. every load RESETS the global
RNGs to 1986. Seed AFTER loading the model if you need a --seed-controlled run.

Example (GPU smoke test):
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  python -u <repo>/cmi_dpo/cosy.py --selftest --out_wav <repo>/logs/cosy_selftest.wav
  (the prompt defaults to a SEAME devman utterance under CMI_DPO_DATA_ROOT; pass
   --prompt_wav/--prompt_text for any other 16 kHz prompt.)
"""
from __future__ import annotations

import argparse
import functools
import inspect
import logging
import os
import sys
import uuid
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

try:
    from . import paths
except ImportError:  # executed as a script (python cmi_dpo/cosy.py --selftest): no parent package
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cmi_dpo import paths

# Default SEAME devman prompt of the --selftest (relative to CMI_DPO_DATA_ROOT; same utterance as
# the original infer.py) and its transcript, plus the synthesised text.
SELFTEST_PROMPT_REL = 'devman/data/format.1/nc12m-06nc12may_0101-000460-000718.flac'
SELFTEST_PROMPT_TEXT = 'okay 好 你 你 介 绍 先 啦'
SELFTEST_TEXT = 'oh wednesday 是 我 们 的 lab 才 ah'

# Sentinel for "use the configured SFT checkpoint" (distinct from None = stock llm.pt).
_CONFIGURED = object()

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# locations (lazy: nothing here needs the CMI_DPO_* environment variables at import time)
# --------------------------------------------------------------------------------------
def cosy_root() -> str:
    """CosyVoice code dir (CMI_DPO_COSY_ROOT; SystemExit with a hint when unset)."""
    return paths.cosy_root()


def cosy_model_dir() -> str:
    """CosyVoice2-0.5B model dir (CMI_DPO_COSY_MODEL_DIR; SystemExit with a hint when unset)."""
    return paths.cosy_model_dir()


def sft_llm_ckpt() -> str:
    """Configured stage-1 SFT LLM checkpoint (CMI_DPO_SFT_CKPT), '' when none is configured."""
    return paths.sft_ckpt()


_LAZY_NAMES = {'COSY_ROOT': cosy_root, 'COSY_MODEL_DIR': cosy_model_dir, 'SFT_LLM_CKPT': sft_llm_ckpt}


def __getattr__(name: str) -> Any:
    """Module-level ``COSY_ROOT`` / ``COSY_MODEL_DIR`` / ``SFT_LLM_CKPT`` resolved on access (PEP 562).

    Keeps ``cosy.COSY_MODEL_DIR`` etc. working for the scripts while nothing is evaluated at
    import time. ``SFT_LLM_CKPT`` is '' when CMI_DPO_SFT_CKPT is empty (= stock llm.pt).
    """
    try:
        return _LAZY_NAMES[name]()
    except KeyError:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from None


def _ensure_cosy_path() -> None:
    """Put the CosyVoice repo and its Matcha-TTS submodule on sys.path (idempotent)."""
    root = cosy_root()
    for path in (root, os.path.join(root, 'Matcha-TTS')):
        if path not in sys.path:
            sys.path.insert(0, path)


def _resolve_llm_ckpt(llm_ckpt: Any) -> str | None:
    """Map a loader's ``llm_ckpt`` argument to a path or None (stock llm.pt)."""
    if llm_ckpt is _CONFIGURED:
        return sft_llm_ckpt() or None
    return llm_ckpt or None


def _model_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def load_llm_ckpt(llm: torch.nn.Module, ckpt: str) -> None:
    """Load a Qwen2LM state_dict saved by train.py (DDP, 'module.' prefix) or by our own
    trainers (no prefix) into ``llm`` with strict=True."""
    sd = torch.load(ckpt, map_location='cpu')
    prefix = 'module.'
    sd = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}
    llm.load_state_dict(sd, strict=True)
    log.info('loaded LLM checkpoint %s (%d tensors)', ckpt, len(sd))


def _construct_cosyvoice2(model_dir: str, device: torch.device) -> Any:
    """``CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False[, device=device])``.

    The CosyVoice checkout used for the paper accepts ``device=``; a fork whose ``__init__``
    lacks that keyword is constructed without it (it then picks cuda:0 / cpu itself) and a
    warning is logged when that choice may differ from the requested device.
    """
    from cosyvoice.cli.cosyvoice import CosyVoice2

    kwargs: dict[str, Any] = {'load_jit': False, 'load_trt': False, 'fp16': False}
    try:
        accepts_device = 'device' in inspect.signature(CosyVoice2.__init__).parameters
    except (TypeError, ValueError):
        accepts_device = True
    if accepts_device:
        return CosyVoice2(model_dir, device=device, **kwargs)
    log.warning('this CosyVoice2 fork takes no device= argument; constructing without it '
                '(requested %s, the fork chooses cuda:0/cpu itself)', device)
    return CosyVoice2(model_dir, **kwargs)


def load_cosyvoice2(device: torch.device | str, llm_ckpt: str | None = _CONFIGURED) -> Any:
    """Full CosyVoice2 (frontend + LLM + flow + hift) on ``device`` with fp16=False, no jit/trt.

    ``llm_ckpt`` replaces the stock llm.pt weights. Default: the configured stage-1 SFT model
    (CMI_DPO_SFT_CKPT), or the stock CosyVoice2 LLM when that variable is empty; pass None to
    force the stock LLM. Global RNGs are reset to 1986 by the yaml.
    """
    _ensure_cosy_path()
    model_dir = cosy_model_dir()
    llm_ckpt = _resolve_llm_ckpt(llm_ckpt)
    device = torch.device(device)
    log.info('loading CosyVoice2 from %s on %s', model_dir, device)
    cv = _construct_cosyvoice2(model_dir, device)
    if llm_ckpt:
        load_llm_ckpt(cv.model.llm, llm_ckpt)
    cv.model.llm.eval()
    return cv


def load_llm_only(device: torch.device | str, llm_ckpt: str | None = _CONFIGURED) -> Any:
    """Lean construction of the Qwen2LM alone (no frontend / onnx sessions), for DPO ranks.

    Mirrors CosyVoice2.__init__ + CosyVoice2Model.load for the LLM part: hyperpyyaml with the
    same overrides, stock llm.pt loaded with strict=False, then ``llm_ckpt`` strict=True.
    ``llm_ckpt`` default: the configured stage-1 SFT model (CMI_DPO_SFT_CKPT), or the stock
    CosyVoice2 LLM when that variable is empty; pass a DPO checkpoint explicitly, or None to
    keep the stock LLM (the initialisation of the stage-1 SFT).
    flow/hift are instantiated on CPU by the yaml and dropped (``del cfg``).
    Global RNGs are reset to 1986 by the yaml.
    """
    _ensure_cosy_path()
    from hyperpyyaml import load_hyperpyyaml

    model_dir = cosy_model_dir()
    llm_ckpt = _resolve_llm_ckpt(llm_ckpt)
    device = torch.device(device)
    log.info('loading Qwen2LM only from %s on %s', model_dir, device)
    with open(os.path.join(model_dir, 'cosyvoice.yaml'), 'r') as f:
        cfg = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN'),
                                             'device': str(device)})
    llm = cfg['llm']
    del cfg
    llm.load_state_dict(torch.load(os.path.join(model_dir, 'llm.pt'), map_location=device), strict=False)
    llm.to(device)
    if llm_ckpt:
        load_llm_ckpt(llm, llm_ckpt)
    llm.eval()
    return llm


def configure_sampling(llm: torch.nn.Module, top_p: float = 0.8, top_k: int = 25,
                       win_size: int = 10, tau_r: float = 0.1) -> None:
    """Rebind ``llm.sampling`` to RAS sampling with explicit hyper-parameters.

    The yaml binds ras_sampling(top_p=0.8, top_k=25, win_size=10, tau_r=0.1); this lets the
    generation scripts honour --top_p / --top_k without editing the yaml.
    """
    _ensure_cosy_path()
    from cosyvoice.utils.common import ras_sampling

    llm.sampling = functools.partial(ras_sampling, top_p=top_p, top_k=top_k, win_size=win_size, tau_r=tau_r)
    log.info('sampler set to RAS top_p=%.3f top_k=%d win_size=%d tau_r=%.3f', top_p, top_k, win_size, tau_r)


# --------------------------------------------------------------------------------------
# prompt / conditioning inputs
# --------------------------------------------------------------------------------------
def build_prompt_inputs(cv: Any, text_tts: str, prompt_text_tts: str, prompt_wav: str,
                        text_frontend: bool = False) -> dict:
    """Build the frontend_zero_shot dict manually (frontend.py) from raw text_tts strings.

    Uses frontend._extract_text_token (plain Qwen tokenizer, no ttsfrd normalisation) unless
    ``text_frontend`` is True, in which case both texts first pass through
    frontend.text_normalize(text, split=False, text_frontend=True).
    Returns the same keys as frontend_zero_shot (tensors on the frontend device, prompt
    tokens/feat trimmed so feat == 2 * token as CosyVoice2 requires) plus
      'text_ids'        LongTensor [Lc] cpu  = prompt_text ids ⊕ target text ids
      'text_len_target' int L               = number of target text ids
      'prompt_speech'   LongTensor [P] cpu  = llm prompt speech tokens
    """
    _ensure_cosy_path()
    import torchaudio
    from cosyvoice.utils.file_utils import load_wav

    fe = cv.frontend
    if text_frontend:
        text_tts = fe.text_normalize(text_tts, split=False, text_frontend=True)
        prompt_text_tts = fe.text_normalize(prompt_text_tts, split=False, text_frontend=True)

    prompt_speech_16k = load_wav(prompt_wav, 16000)  # [1, T] float cpu
    prompt_dur = prompt_speech_16k.shape[1] / 16000.0
    if prompt_dur > 30.0:
        raise ValueError(f'prompt {prompt_wav} is {prompt_dur:.2f} s > 30 s (speech tokenizer limit)')

    tts_text_token, tts_text_token_len = fe._extract_text_token(text_tts)
    if tts_text_token.shape[1] < 1:
        raise ValueError(f'target text tokenises to 0 tokens: {text_tts!r}')
    prompt_text_token, prompt_text_token_len = fe._extract_text_token(prompt_text_tts)

    prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=cv.sample_rate)(prompt_speech_16k)
    speech_feat, speech_feat_len = fe._extract_speech_feat(prompt_speech_resample)
    speech_token, speech_token_len = fe._extract_speech_token(prompt_speech_16k)
    if cv.sample_rate == 24000:
        # cosyvoice2, force speech_feat % speech_token = 2 (frontend_zero_shot)
        token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
        speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
        speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
    embedding = fe._extract_spk_embedding(prompt_speech_16k)

    model_input = {'text': tts_text_token, 'text_len': tts_text_token_len,
                   'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
                   'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                   'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                   'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                   'llm_embedding': embedding, 'flow_embedding': embedding}
    model_input['text_ids'] = torch.cat([prompt_text_token, tts_text_token], dim=1).squeeze(0).long().cpu()
    model_input['text_len_target'] = int(tts_text_token.shape[1])
    model_input['prompt_speech'] = speech_token.squeeze(0).long().cpu()
    return model_input


# --------------------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------------------
@torch.inference_mode()
def generate_speech_tokens_ex(llm: torch.nn.Module, inputs: dict, temperature: float = 1.0, sampling: int = 25,
                              min_token_text_ratio: float = 2, max_token_text_ratio: float = 20
                              ) -> tuple[list[int], bool]:
    """Sample one speech-token sequence for ``inputs``; returns (tokens, ended_with_eos).

    Re-implementation of Qwen2LM.inference's loop (cosyvoice/llm/llm.py) with two deliberate
    deviations:
      1. logits are divided by ``temperature`` before log_softmax;
      2. the fill ids > speech_token_size (6562/6563, never training targets) are masked to -inf
         BEFORE sampling instead of being skipped with ``continue`` afterwards. Upstream's
         ``continue`` leaves lm_input stale, so the next forward_one_step re-feeds the previous
         input (the whole prefix when i == 0) and the KV cache gains a duplicated position that
         the clean [sos, text, task, prompt ⊕ cand] layout scored by sequence_logps does not
         contain; it also burned one max_len iteration. Masking keeps the generated context
         identical to the scored one. (RAS's random_sampling fallback samples over all 6564
         classes, so the fill ids really can be drawn without the mask.)
    min/max lengths use the TARGET text length only (text_len - prompt_text_len), EOS is
    re-sampled away while i < min_len (llm.sampling_ids ignore_eos), EOS stops.
    ``ended_with_eos`` is False when the loop ran out of max_len iterations without sampling
    EOS (the candidate was truncated); DPO scoring should then not add the EOS term
    (sequence_logps ``eos_mask``). Returns the generated ids (all < speech_token_size).
    """
    if temperature <= 0:
        raise ValueError(f'temperature must be > 0, got {temperature}')
    device = _model_device(llm)
    text = inputs['text'].to(device)
    prompt_text = inputs['prompt_text'].to(device)
    prompt_speech_token = inputs['llm_prompt_speech_token'].to(device)
    target_text_len = int(text.shape[1])  # == text_len - prompt_text_len in Qwen2LM.inference

    text = torch.concat([prompt_text, text], dim=1)

    # 1. encode text
    text = llm.llm.model.model.embed_tokens(text)

    # 2. encode embedding (Qwen2LM uses no speaker embedding in the LLM)
    embedding = torch.zeros(1, 0, llm.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)

    # 3. concat llm_input
    sos_eos_emb = llm.llm_embedding.weight[llm.sos_eos].reshape(1, 1, -1)
    task_id_emb = llm.llm_embedding.weight[llm.task_id].reshape(1, 1, -1)
    if prompt_speech_token.shape[1] != 0:
        prompt_speech_token_emb = llm.speech_embedding(prompt_speech_token)
    else:
        prompt_speech_token_emb = torch.zeros(1, 0, llm.llm_input_size, dtype=text.dtype).to(device)
    lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

    # 4. cal min/max_length
    min_len = int(target_text_len * min_token_text_ratio)
    max_len = int(target_text_len * max_token_text_ratio)

    # 5. step by step decode
    out_tokens: list[int] = []
    ended_with_eos = False
    cache = None
    for i in range(max_len):
        y_pred, cache = llm.llm.forward_one_step(lm_input,
                                                 masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                                                             device=lm_input.device)).to(torch.bool),
                                                 cache=cache)
        logits = llm.llm_decoder(y_pred[:, -1]) / temperature
        logits[:, llm.speech_token_size + 1:] = -float('inf')  # never draw the fill ids (see docstring)
        logp = logits.log_softmax(dim=-1)
        top_ids = llm.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
        if top_ids == llm.speech_token_size:
            ended_with_eos = True
            break
        if top_ids > llm.speech_token_size:  # unreachable after the mask; kept as a guard
            raise RuntimeError(f'generate_speech_tokens: sampled masked fill id {top_ids}')
        out_tokens.append(top_ids)
        lm_input = llm.speech_embedding.weight[top_ids].reshape(1, 1, -1)
    return out_tokens, ended_with_eos


def generate_speech_tokens(llm: torch.nn.Module, inputs: dict, temperature: float = 1.0, sampling: int = 25,
                           min_token_text_ratio: float = 2, max_token_text_ratio: float = 20) -> list[int]:
    """Convenience wrapper around generate_speech_tokens_ex that returns the tokens only.

    A returned list of exactly ``int(text_len_target * max_token_text_ratio)`` tokens means the
    sequence was truncated (no EOS); use generate_speech_tokens_ex to get the flag directly.
    """
    tokens, _ = generate_speech_tokens_ex(llm, inputs, temperature=temperature, sampling=sampling,
                                          min_token_text_ratio=min_token_text_ratio,
                                          max_token_text_ratio=max_token_text_ratio)
    return tokens


def tokens_to_wav(cv: Any, tokens: list[int], inputs: dict) -> torch.Tensor:
    """Vocode ``tokens`` with the prompt conditioning in ``inputs`` via cv.model.token2wav.

    Non-streaming call (token_offset=0, finalize=True, speed=1.0) with a fresh uuid key in
    hift_cache_dict, set to None before and popped after. token2wav prints ``type(tts_mel)``
    once (upstream code); harmless. Returns float32 cpu [1, S] at cv.sample_rate (24 kHz).
    """
    if len(tokens) == 0:
        raise ValueError('tokens_to_wav: empty token list')
    model = cv.model
    key = str(uuid.uuid4())
    token = torch.tensor([tokens], dtype=torch.int32, device=model.device)
    model.hift_cache_dict[key] = None
    try:
        with torch.no_grad():
            wav = model.token2wav(token=token,
                                  prompt_token=inputs['flow_prompt_speech_token'],
                                  prompt_feat=inputs['prompt_speech_feat'],
                                  embedding=inputs['flow_embedding'],
                                  uuid=key, token_offset=0, finalize=True, speed=1.0)
    finally:
        model.hift_cache_dict.pop(key, None)
    return wav.detach().float().cpu()


# --------------------------------------------------------------------------------------
# scoring (DPO)
# --------------------------------------------------------------------------------------
def sequence_logps(llm: torch.nn.Module,
                   text_ids: torch.Tensor, text_lens: torch.Tensor,
                   prompt_speech: torch.Tensor, prompt_lens: torch.Tensor,
                   cand: torch.Tensor, cand_lens: torch.Tensor,
                   include_eos: bool = True, return_lengths: bool = False,
                   eos_mask: torch.Tensor | None = None):
    """Sum of token log-probs of each candidate under ``llm`` in the INFERENCE layout.

    Args (all right-padded, any device; moved to the model's device):
      text_ids [B, Lc] long, text_lens [B]        prompt_text ⊕ text ids (as generated)
      prompt_speech [B, P] long, prompt_lens [B]  llm prompt speech tokens
      cand [B, T] long, cand_lens [B]             candidate speech tokens (< speech_token_size)
      include_eos                                 add log p(EOS | ..., cand[T-1]) for every item
      eos_mask [B] bool (optional)                per-item override of include_eos: True adds the
                                                  EOS term, False does not. Use the ended_with_eos
                                                  flag of generate_speech_tokens_ex so that a
                                                  candidate cut by max_len (no EOS ever sampled) is
                                                  not scored on an event the sampler never produced.
      return_lengths                              also return the number of scored tokens [B]
    Returns Tensor [B] (float32) — or (logps [B], lengths [B]) when return_lengths=True.
    No temperature: the model's own distribution is scored. Gradients flow when enabled
    (this is the DPO policy path); wrap in torch.no_grad() for the frozen reference.

    Derivation of ``start`` (index whose logits predict cand[0]):
      Qwen2LM.forward builds lm_input = [sos, embed(text) (L), task, speech_emb(speech) (T)] and
      lm_target = [IGNORE]*(1+L) + speech + [EOS]; LabelSmoothingLoss compares logits[:, i] with
      lm_target[:, i], so the logits at index 1+L (the task token) predict speech[0], the logits at
      the position of speech[t-1] predict speech[t], and the logits at speech[T-1] predict EOS.
      Qwen2LM.inference feeds [sos, embed(prompt_text ⊕ text) (Lc), task, speech_emb(prompt_speech) (P)]
      and continues the speech stream with the generated tokens, so the P prompt tokens occupy
      speech[0..P-1] and cand[0] is predicted at the position of the last prompt token:
      (1 + Lc + 1 + P) - 1 = 1 + Lc + P (== the task index 1+Lc when P == 0). cand[t] is predicted
      at start+t and EOS at start+T; the T+1 scored positions start..start+T end exactly at the
      last valid position of the sequence (length 1 + Lc + 1 + P + T = start + T + 1).
    """
    device = _model_device(llm)
    text_ids = text_ids.to(device).long()
    prompt_speech = prompt_speech.to(device).long()
    cand = cand.to(device).long()
    text_lens_l = [int(x) for x in text_lens.tolist()]
    prompt_lens_l = [int(x) for x in prompt_lens.tolist()]
    cand_lens_l = [int(x) for x in cand_lens.tolist()]
    batch_size = text_ids.shape[0]
    eos = llm.speech_token_size
    if eos_mask is None:
        eos_flags = [bool(include_eos)] * batch_size
    else:
        eos_flags = [bool(x) for x in eos_mask.reshape(-1).tolist()]
        if len(eos_flags) != batch_size:
            raise ValueError(f'sequence_logps: eos_mask has {len(eos_flags)} entries for batch of {batch_size}')

    embed_text = llm.llm.model.model.embed_tokens
    sos_eos_emb = llm.llm_embedding.weight[llm.sos_eos].reshape(1, -1)
    task_id_emb = llm.llm_embedding.weight[llm.task_id].reshape(1, -1)

    seqs: list[torch.Tensor] = []
    starts: list[int] = []
    targets: list[torch.Tensor] = []
    n_scored: list[int] = []
    for b in range(batch_size):
        lc, p, t = text_lens_l[b], prompt_lens_l[b], cand_lens_l[b]
        if t < 1:
            raise ValueError(f'sequence_logps: item {b} has an empty candidate')
        speech_ids = torch.cat([prompt_speech[b, :p], cand[b, :t]], dim=0)
        x = torch.cat([sos_eos_emb, embed_text(text_ids[b, :lc]), task_id_emb, llm.speech_embedding(speech_ids)], dim=0)
        seqs.append(x)
        starts.append(1 + lc + p)
        targets.append(torch.cat([cand[b, :t], cand.new_tensor([eos])], dim=0))
        n_scored.append(t + 1 if eos_flags[b] else t)

    # like Qwen2LM.pad_unpad_sequence: right-pad with zeros + attention_mask of valid positions
    lm_input = pad_sequence(seqs, batch_first=True, padding_value=0.0)  # [B, S, D]
    seq_len = lm_input.shape[1]
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    for b, x in enumerate(seqs):
        attention_mask[b, :x.shape[0]] = 1

    # Inner Qwen2Model == Qwen2ForCausalLM.hidden_states[-1] (post-norm, transformers 4.40.1
    # modeling_qwen2.py:1071-1075) without the 151936-way lm_head that Qwen2LM.forward computes
    # and discards.
    hidden = llm.llm.model.model(inputs_embeds=lm_input, attention_mask=attention_mask,
                                 use_cache=False, return_dict=True).last_hidden_state  # [B, S, D]

    t_max = max(n_scored)
    pos = torch.zeros(batch_size, t_max, dtype=torch.long, device=device)
    tgt = torch.zeros(batch_size, t_max, dtype=torch.long, device=device)
    mask = torch.zeros(batch_size, t_max, dtype=torch.bool, device=device)
    for b in range(batch_size):
        n = n_scored[b]
        pos[b, :n] = torch.arange(starts[b], starts[b] + n, device=device)
        tgt[b, :n] = targets[b][:n]
        mask[b, :n] = True

    h = hidden.gather(1, pos.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))  # [B, t_max, D]
    logp = llm.llm_decoder(h).float().log_softmax(dim=-1)                       # [B, t_max, V]
    tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)                   # [B, t_max]
    seq_logp = tok_logp.masked_fill(~mask, 0.0).sum(dim=-1)                    # [B]
    if return_lengths:
        return seq_logp, torch.tensor(n_scored, dtype=torch.long, device=device)
    return seq_logp


# --------------------------------------------------------------------------------------
# self-test (GPU)
# --------------------------------------------------------------------------------------
def _selftest(args: argparse.Namespace) -> None:
    _ensure_cosy_path()
    import torchaudio
    from cosyvoice.utils.common import set_all_random_seed

    prompt_wav = args.prompt_wav
    if not prompt_wav:
        data_root = paths.data_root()
        if not data_root:
            raise SystemExit('--prompt_wav not given and CMI_DPO_DATA_ROOT is unset: pass --prompt_wav '
                             '(+ --prompt_text) or export CMI_DPO_DATA_ROOT so the default '
                             f'SEAME prompt <DATA_ROOT>/{SELFTEST_PROMPT_REL} can be used')
        prompt_wav = os.path.join(data_root, SELFTEST_PROMPT_REL)
    if not os.path.isfile(prompt_wav):
        raise SystemExit(f'prompt wav not found: {prompt_wav}')

    device = torch.device(args.device)
    cv = load_cosyvoice2(device, llm_ckpt=args.ckpt or None)
    set_all_random_seed(args.seed)  # after loading: the yaml reseeds to 1986
    llm = cv.model.llm

    inputs = build_prompt_inputs(cv, args.text, args.prompt_text, prompt_wav,
                                 text_frontend=args.text_frontend)
    log.info('inputs: text %s prompt_text %s prompt_speech %s feat %s text_ids %s text_len_target %d',
             tuple(inputs['text'].shape), tuple(inputs['prompt_text'].shape),
             tuple(inputs['llm_prompt_speech_token'].shape), tuple(inputs['prompt_speech_feat'].shape),
             tuple(inputs['text_ids'].shape), inputs['text_len_target'])

    tokens, ended_with_eos = generate_speech_tokens_ex(llm, inputs, temperature=args.temperature)
    log.info('generated %d speech tokens (min %s max %s, ended_with_eos=%s): %s ...', len(tokens),
             min(tokens) if tokens else None, max(tokens) if tokens else None, ended_with_eos, tokens[:16])

    wav = tokens_to_wav(cv, tokens, inputs)
    log.info('wav shape %s dtype %s -> %.2f s at %d Hz', tuple(wav.shape), wav.dtype,
             wav.shape[1] / cv.sample_rate, cv.sample_rate)
    if args.out_wav:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_wav)), exist_ok=True)
        torchaudio.save(args.out_wav, wav, cv.sample_rate)
        log.info('saved %s', args.out_wav)

    cand = torch.tensor(tokens, dtype=torch.long)
    text_ids = inputs['text_ids']
    prompt_speech = inputs['prompt_speech']
    with torch.no_grad():
        lp1, n1 = sequence_logps(llm, text_ids.unsqueeze(0), torch.tensor([text_ids.numel()]),
                                 prompt_speech.unsqueeze(0), torch.tensor([prompt_speech.numel()]),
                                 cand.unsqueeze(0), torch.tensor([cand.numel()]), include_eos=True, return_lengths=True,
                                 eos_mask=torch.tensor([ended_with_eos]))
        # padding check: batch the full candidate with a truncated copy; item 0 must reproduce lp1
        half = max(1, cand.numel() // 2)
        cand_b = pad_sequence([cand, cand[:half]], batch_first=True, padding_value=0)
        lp2 = sequence_logps(llm, text_ids.unsqueeze(0).expand(2, -1), torch.tensor([text_ids.numel()] * 2),
                             prompt_speech.unsqueeze(0).expand(2, -1), torch.tensor([prompt_speech.numel()] * 2),
                             cand_b, torch.tensor([cand.numel(), half]), include_eos=True)
    log.info('sequence_logps single: %.4f over %d scored tokens (%.4f / token)',
             lp1.item(), int(n1.item()), lp1.item() / int(n1.item()))
    log.info('sequence_logps batched [full, half]: %s ; |batched[0] - single| = %.3e',
             [round(v, 4) for v in lp2.tolist()], abs(lp2[0].item() - lp1.item()))
    print(f'COSY_SELFTEST_OK tokens={len(tokens)} wav={tuple(wav.shape)} logp={lp1.item():.4f} '
          f'scored={int(n1.item())} batch_diff={abs(lp2[0].item() - lp1.item()):.3e}', flush=True)


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selftest', action='store_true',
                    help='load CosyVoice2 on --device, build inputs for the prompt, generate one candidate, '
                         'vocode it and score it with sequence_logps')
    ap.add_argument('--device', default='cuda:0', help='torch device for the self-test (default cuda:0)')
    ap.add_argument('--ckpt', default=sft_llm_ckpt(),
                    help='LLM checkpoint to load (default: CMI_DPO_SFT_CKPT); empty string keeps the stock LLM')
    ap.add_argument('--prompt_wav', default='',
                    help='16 kHz prompt wav (default: <CMI_DPO_DATA_ROOT>/' + SELFTEST_PROMPT_REL +
                         '; an error is raised when DATA_ROOT is unset)')
    ap.add_argument('--prompt_text', default=SELFTEST_PROMPT_TEXT,
                    help='transcript of --prompt_wav in SEAME spacing (default: that of the default prompt)')
    ap.add_argument('--text', default=SELFTEST_TEXT, help='text to synthesise (SEAME spacing)')
    ap.add_argument('--seed', type=int, default=0, help='seed applied after model loading (default 0)')
    ap.add_argument('--temperature', type=float, default=1.0, help='sampling temperature (default 1.0)')
    ap.add_argument('--text_frontend', action='store_true',
                    help='apply frontend.text_normalize(split=False, text_frontend=True) before tokenising')
    ap.add_argument('--out_wav', default='', help='optional path to save the self-test 24 kHz wav')
    return ap


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    parser = _build_argparser()
    cli = parser.parse_args()
    if not cli.selftest:
        parser.print_help()
        sys.exit(0)
    _selftest(cli)
