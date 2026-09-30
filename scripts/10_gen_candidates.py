#!/usr/bin/env python3
"""Candidate generation for multi-critic DPO (paper section 3.2.1; see docs/DESIGN.md).

For every manifest row (utt, text_tts, prompt_utt / prompt_wav / prompt_text) this script builds
the CosyVoice2 zero-shot conditioning ONCE (prompt speech tokens, mel feat, speaker embedding,
text ids), then draws N speech-token candidates from the fine-tuned Qwen2LM with RAS sampling at
temperature tau, vocodes each candidate with flow + HiFT, and writes

    <out>/wav/<utt>/<k>.wav        24 kHz, k = 0..N-1     (--save_16k also writes <k>_16k.wav)
    <out>/tokens/<utt>.pt          {'utt', 'text_ids', 'text_len_target', 'prompt_speech',
                                    'cands': [LongTensor]*N, 'ended_with_eos': [bool]*N, 'prompt_utt'}
                                   (ended_with_eos[k] False = candidate k was cut by max_len without
                                    sampling EOS; 14/15 then score it without the EOS term)
    <out>/cands.tsv                utt  cand  wav  dur  n_tokens      (append-only)
    <out>/failed.txt               utt <TAB> reason                    (append-only)
    <out>/gen_config.json          config fingerprint (see below)

Text: text_tts and prompt_text are taken from the manifest VERBATIM. 00_prep_manifest.py applied
--text_mode (strip | raw) to both columns, so nothing is re-stripped here; the mode is inferred
from the manifest (rows whose text_raw carries <...> tags) and recorded in the fingerprint.

Config fingerprint: <out>/gen_config.json stores the checkpoint (path, mtime, size), temperature,
n_cand, sampling, top_p/top_k (null = model default), seed, text_frontend, text_mode and
manifest_sha256 (common.manifest_content_hash: an order-independent hash of the utt / text_raw /
text_tts / prompt_utt / prompt_wav / prompt_text columns, so a manifest re-generated at the same
path with another prompt assignment, --seed or --text_mode is detected; the manifest PATH and row
count are stored for information only). On a resume (existing outputs) the current config is
compared with the stored one and the run ABORTS on any difference unless --force_resume is given
(then the differences are logged and the fingerprint is overwritten). A stored fingerprint that
lacks a newer key (e.g. manifest_sha256 from a run before it existed) cannot be checked for that
key: it only warns and the fingerprint is re-written with the current values. Outputs that predate
the fingerprint (no gen_config.json) only produce a warning and get one written. All shards of one
run write the same fingerprint.

Sharding: with --nshards > 1 each shard appends to ITS OWN <out>/cands.tsv.<shard> and
<out>/failed.txt.<shard> (one writer per file: NFS O_APPEND is not atomic across nodes and the
header would race), then `--merge` once afterwards (no GPU) concatenates the shard files (plus any
existing cands.tsv), de-duplicates on (utt, cand), drops rows whose tokens/<utt>.pt is missing
(interrupted utterance) and writes the final <out>/cands.tsv that 11/12/14 read. With --nshards 1
the final files are written directly.
Resumable: an utterance whose <out>/tokens/<utt>.pt exists is skipped (the tokens file is written
last, so its presence means wavs + cands.tsv rows are complete); rows already present in this
shard's cands file are never appended twice. Deterministic: candidate k of utterance u is drawn
after re-seeding all RNGs from sha256(seed, u, k, attempt), so every shard reproduces the same
candidates regardless of --shard/--nshards or restart point.
--dry_run builds the fingerprint and the plan (sharded/limited rows with their done/pending
status) WITHOUT loading the model, prints both and exits 0; a fingerprint mismatch aborts the dry
run exactly like a real run would.
Exit status is 1 when every attempted utterance failed (so downstream steps that require exit
status 0 stop).

Environment: one GPU, the CosyVoice env (torch 2.3.1+cu121),
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1.
Paths: the CosyVoice2 checkout / model dir and the default --ckpt (CMI_DPO_SFT_CKPT) come from
cmi_dpo.paths (the CMI_DPO_* environment variables); --show_paths prints them and exits.

Example:
    python -u scripts/10_gen_candidates.py \
        --manifest data/train_100h.tsv \
        --out exp/round1/cands \
        --n_cand 4 --temperature 1.0 --seed 0 --shard 0 --nshards 6
    python -u scripts/10_gen_candidates.py --out exp/round1/cands --merge
"""
from __future__ import annotations

import argparse
import functools
import glob
import hashlib
import json
import logging
import os
import socket
import sys
import time
from datetime import datetime
from typing import Any

import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmi_dpo import common  # noqa: E402
from cmi_dpo import cosy  # noqa: E402
from cmi_dpo import paths  # noqa: E402

CANDS_COLUMNS = ['utt', 'cand', 'wav', 'dur', 'n_tokens']
SAMPLE_RATE = 24000
MAX_SPEECH_TOKEN_ID = 6561  # EOS id; the vocoder never sees it
CONFIG_NAME = 'gen_config.json'
CONFIG_KEYS = ('ckpt', 'ckpt_file', 'ckpt_mtime', 'ckpt_size', 'temperature', 'n_cand', 'sampling', 'top_p', 'top_k',
               'seed', 'text_frontend', 'text_mode', 'manifest_sha256')

logger = logging.getLogger('gen_candidates')


class ShowPathsAction(argparse.Action):
    """--show_paths: print cmi_dpo.paths.describe() and exit, before the required flags are checked."""

    def __init__(self, option_strings: list[str], dest: str, **kwargs: Any) -> None:
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS,
                         help=kwargs.get('help', 'print the configured paths (cmi_dpo.paths) and exit'))

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
                 values: Any, option_string: str | None = None) -> None:
        print(paths.describe())
        parser.exit()


def resolve_ckpt(arg: str | None) -> str | None:
    """--ckpt value -> LLM checkpoint path, or None for the stock llm.pt.

    Flag omitted (None) = the configured CMI_DPO_SFT_CKPT (None when it is unset / empty);
    'none' (case-insensitive) or '' = the stock CosyVoice2 llm.pt; anything else = that path.
    Resolved after parsing so that --help works without the CMI_DPO_* environment variables.
    """
    if arg is None:
        return paths.sft_ckpt() or None
    return None if arg.strip().lower() in ('', 'none') else arg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Generate N sampled CosyVoice2 candidates per manifest utterance (DPO candidate pool).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--manifest', default=None, help='manifest TSV from 00_prep_manifest.py (required unless --merge)')
    p.add_argument('--out', required=True, help='output dir (wav/, tokens/, cands.tsv, failed.txt, gen_config.json)')
    p.add_argument('--ckpt', default=None,
                   help='fine-tuned Qwen2LM state_dict (stage-1 SFT or DPO checkpoint); default = the configured '
                        'CMI_DPO_SFT_CKPT (stock llm.pt when unset); "none" = stock llm.pt')
    p.add_argument('--n_cand', type=int, default=4, help='candidates per utterance (N)')
    p.add_argument('--temperature', type=float, default=1.0, help='softmax temperature on LLM logits')
    p.add_argument('--sampling', type=int, default=25,
                   help='`sampling` argument of Qwen2LM.sampling_ids (k of the RAS random fallback)')
    p.add_argument('--top_p', type=float, default=None,
                   help='override RAS nucleus top_p (model default 0.8 from cosyvoice.yaml)')
    p.add_argument('--top_k', type=int, default=None,
                   help='override RAS nucleus top_k (model default 25 from cosyvoice.yaml)')
    p.add_argument('--seed', type=int, default=0, help='base seed; per-candidate seeds are derived from it')
    p.add_argument('--shard', type=int, default=0, help='this shard index (0-based)')
    p.add_argument('--nshards', type=int, default=1, help='number of shards the manifest is split into')
    p.add_argument('--limit', type=int, default=0, help='process at most this many rows of the shard (0 = all)')
    p.add_argument('--save_16k', action='store_true', help='also write <k>_16k.wav next to each 24 kHz wav')
    p.add_argument('--text_frontend', action='store_true',
                   help='apply frontend.text_normalize (ttsfrd) before tokenising; default keeps SEAME text as-is')
    p.add_argument('--max_retries', type=int, default=2,
                   help='extra sampling attempts (fresh seed) when a candidate hits the sampler max_trials error')
    p.add_argument('--force_resume', action='store_true',
                   help='resume even when <out>/gen_config.json differs from the current config (fingerprint is overwritten)')
    p.add_argument('--dry_run', action='store_true',
                   help='print the fingerprint + the planned rows (no model load, nothing written) and exit 0')
    p.add_argument('--merge', action='store_true',
                   help='no generation: merge <out>/cands.tsv.<shard> (+ failed.txt.<shard>) into the final files')
    p.add_argument('--device', default='cuda:0', help='torch device')
    p.add_argument('--show_paths', action=ShowPathsAction)
    return p.parse_args()


def derive_seed(base: int, utt: str, k: int, attempt: int) -> int:
    """Stable 31-bit seed for candidate k / attempt of utterance utt under base seed."""
    h = hashlib.sha256(f'{base}|{utt}|{k}|{attempt}'.encode('utf-8')).digest()
    return int.from_bytes(h[:4], 'big') & 0x7FFFFFFF


def override_sampler(llm: torch.nn.Module, top_p: float | None, top_k: int | None) -> None:
    """Rebind llm.sampling (a functools.partial of ras_sampling) with new top_p / top_k."""
    if top_p is None and top_k is None:
        return
    sampler = llm.sampling
    if not isinstance(sampler, functools.partial):
        raise TypeError(f'llm.sampling is {type(sampler)}, expected functools.partial; cannot override top_p/top_k')
    kw = dict(sampler.keywords or {})
    if top_p is not None:
        kw['top_p'] = top_p
    if top_k is not None:
        kw['top_k'] = top_k
    llm.sampling = functools.partial(sampler.func, *sampler.args, **kw)
    logger.info('RAS sampler overridden: %s', kw)


# ---------------------------------------------------------------------------
# config fingerprint
# ---------------------------------------------------------------------------
def derive_text_mode(rows: list[dict]) -> str:
    """Infer the 00_prep_manifest --text_mode from the manifest itself.

    Only rows whose text_raw carries <...> tags discriminate: text_tts == text_raw -> 'raw',
    text_tts == strip_tags(text_raw) -> 'strip'. Returns 'indeterminate' when no row carries a tag
    (both modes coincide) and 'mixed' when rows disagree or match neither convention."""
    n_raw = n_strip = n_other = 0
    for r in rows:
        raw = ' '.join(r['text_raw'].split())
        stripped = common.strip_tags(raw)
        if stripped == raw:
            continue
        tts = ' '.join(r['text_tts'].split())
        if tts == raw:
            n_raw += 1
        elif tts == stripped:
            n_strip += 1
        else:
            n_other += 1
    if n_other or (n_raw and n_strip):
        return 'mixed'
    if n_raw:
        return 'raw'
    if n_strip:
        return 'strip'
    return 'indeterminate'


def ckpt_stats(ckpt: str | None) -> tuple[str, float | None, int | None]:
    """(resolved checkpoint file, mtime, size); the stock llm.pt when ckpt is None. Missing file -> None stats."""
    path = os.path.abspath(ckpt) if ckpt else os.path.join(paths.cosy_model_dir(), 'llm.pt')
    if not os.path.isfile(path):
        return path, None, None
    st = os.stat(path)
    return path, st.st_mtime, st.st_size


def build_config(args: argparse.Namespace, ckpt: str | None, text_mode: str, manifest_rows: list[dict]) -> dict:
    """Fingerprint document: 'config' holds the compared keys (CONFIG_KEYS), the rest is informational.

    manifest_sha256 = common.manifest_content_hash(manifest_rows) (order-independent content hash of
    the conditioning columns); the manifest path / row count outside 'config' are informational."""
    ckpt_file, mtime, size = ckpt_stats(ckpt)
    config = {
        'ckpt': os.path.abspath(ckpt) if ckpt else 'none',
        'ckpt_file': ckpt_file,
        'ckpt_mtime': mtime,
        'ckpt_size': size,
        'temperature': float(args.temperature),
        'n_cand': int(args.n_cand),
        'sampling': int(args.sampling),
        'top_p': args.top_p,
        'top_k': args.top_k,
        'seed': int(args.seed),
        'text_frontend': bool(args.text_frontend),
        'text_mode': text_mode,
        'manifest_sha256': common.manifest_content_hash(manifest_rows),
    }
    assert tuple(config) == CONFIG_KEYS
    return {
        'script': os.path.basename(__file__),
        'created': datetime.now().isoformat(timespec='seconds'),
        'host': socket.gethostname(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'manifest': os.path.abspath(args.manifest),
        'manifest_rows': len(manifest_rows),
        'argv': sys.argv[1:],
        'config': config,
    }


def config_diff(stored: dict, current: dict) -> tuple[dict[str, tuple], list[str]]:
    """(differences, unchecked keys) between the compared 'config' sections.

    differences: key -> (stored, current) for every key the STORED fingerprint has whose value
    differs; unchecked: keys of the current config that the stored one lacks (fingerprint written
    by an older version of the script, e.g. before manifest_sha256 existed) -- they cannot be
    verified and only warrant a warning + a re-written fingerprint."""
    a = stored.get('config', {})
    b = current.get('config', {})
    diff = {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if k in a and a.get(k) != b.get(k)}
    unchecked = [k for k in b if k not in a]
    return diff, unchecked


def outputs_exist(out: str) -> bool:
    """True when the output dir already holds generation outputs (tokens, cands files or wavs)."""
    if not os.path.isdir(out):
        return False
    if glob.glob(os.path.join(out, 'cands.tsv*')) or glob.glob(os.path.join(out, 'tokens', '*.pt')):
        return True
    wav_root = os.path.join(out, 'wav')
    return os.path.isdir(wav_root) and bool(os.listdir(wav_root))


def check_fingerprint(out: str, current: dict, force_resume: bool) -> tuple[dict | None, bool]:
    """Compare the current fingerprint with <out>/gen_config.json.

    Returns (stored fingerprint or None, write_needed). Aborts via SystemExit on a mismatch unless
    force_resume. Existing outputs without a fingerprint (pre-fingerprint runs) only warn."""
    path = os.path.join(out, CONFIG_NAME)
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            stored = json.load(f)
        diff, unchecked = config_diff(stored, current)
        if not diff:
            if unchecked:
                logger.warning('fingerprint %s (created %s by job %s) predates the key(s) %s: they cannot be verified '
                               'against the existing outputs; re-writing the fingerprint with the current values',
                               path, stored.get('created'), stored.get('slurm_job_id'), ', '.join(unchecked))
                return stored, True
            logger.info('fingerprint %s matches the current config (created %s by job %s)', path,
                        stored.get('created'), stored.get('slurm_job_id'))
            return stored, False
        lines = [f'  {k}: stored={a!r} current={b!r}' for k, (a, b) in diff.items()]
        msg = (f'CONFIG MISMATCH: {path} (created {stored.get("created")}) differs from the current run in '
               f'{len(diff)} key(s):\n' + '\n'.join(lines))
        if not force_resume:
            raise SystemExit(msg + '\nRefusing to mix candidates of different configs in one output dir; use a new '
                             '--out, or pass --force_resume to continue anyway (fingerprint is overwritten).')
        logger.warning('%s\n--force_resume given: continuing and overwriting the fingerprint', msg)
        return stored, True
    if outputs_exist(out):
        logger.warning('%s has outputs but no %s (written before fingerprinting): cannot verify the config of the '
                       'existing outputs; writing the current fingerprint', out, CONFIG_NAME)
    return None, True


def write_fingerprint(out: str, doc: dict) -> str:
    path = os.path.join(out, CONFIG_NAME)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write('\n')
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# cands.tsv IO / merge
# ---------------------------------------------------------------------------
def record_failure(path: str, utt: str, reason: str) -> None:
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f'{utt}\t{reason}\n')


def cands_paths(out: str, shard: int, nshards: int) -> tuple[str, str]:
    """(cands.tsv, failed.txt) of this shard: plain names for --nshards 1, else '.<shard>' suffixed."""
    suffix = '' if nshards == 1 else f'.{shard}'
    return os.path.join(out, 'cands.tsv' + suffix), os.path.join(out, 'failed.txt' + suffix)


def read_cands_rows(path: str) -> list[dict[str, str]]:
    """Tolerant cands.tsv reader for resume/merge: header lines are skipped wherever they occur and a
    malformed (torn) line is dropped with a warning instead of raising like common.read_tsv."""
    rows: list[dict[str, str]] = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding='utf-8') as f:
        for ln, line in enumerate(f, start=1):
            line = line.rstrip('\n')
            if not line:
                continue
            vals = line.split('\t')
            if vals == CANDS_COLUMNS:
                continue
            if len(vals) != len(CANDS_COLUMNS):
                logger.warning('%s:%d: malformed line dropped (%d columns, expected %d)', path, ln, len(vals),
                               len(CANDS_COLUMNS))
                continue
            rows.append(dict(zip(CANDS_COLUMNS, vals)))
    return rows


def _cand_key(row: dict) -> tuple[str, str]:
    return row['utt'], str(row['cand'])


def merge_shards(out: str) -> None:
    """Merge <out>/cands.tsv.<shard> (and an existing <out>/cands.tsv) into the final <out>/cands.tsv.

    Rows are de-duplicated on (utt, cand) (duplicates are bit-identical by construction), rows whose
    tokens/<utt>.pt is missing (utterance interrupted before its completion marker) are dropped and
    reported, the result is sorted by (utt, cand) and written atomically. failed.txt.<shard> files are
    merged into <out>/failed.txt likewise (one line per utt, last wins)."""
    final = os.path.join(out, 'cands.tsv')
    parts = sorted(glob.glob(final + '.[0-9]*'))
    if os.path.exists(final):
        parts.insert(0, final)
    if not parts:
        raise SystemExit(f'--merge: no cands.tsv / cands.tsv.<shard> found in {out}')
    tok_root = os.path.join(out, 'tokens')
    merged: dict[tuple[str, str], dict[str, str]] = {}
    n_in = 0
    for part in parts:
        rows = read_cands_rows(part)
        n_in += len(rows)
        for r in rows:
            merged[_cand_key(r)] = r  # later files win; duplicates carry identical content
        logger.info('read %d rows from %s', len(rows), part)
    incomplete = sorted({k[0] for k in merged if not os.path.exists(os.path.join(tok_root, k[0] + '.pt'))})
    if incomplete:
        logger.warning('dropping %d utts without tokens/<utt>.pt (interrupted before completion): %s%s',
                       len(incomplete), ' '.join(incomplete[:10]), ' ...' if len(incomplete) > 10 else '')
    keep = [k for k in merged if k[0] not in set(incomplete)]
    keep.sort(key=lambda k: (k[0], int(k[1]) if k[1].isdigit() else -1, k[1]))
    common.write_tsv(final, [merged[k] for k in keep], CANDS_COLUMNS)
    logger.info('merged %d files, %d rows in -> %d unique rows (%d utts) -> %s', len(parts), n_in, len(keep),
                len({k[0] for k in keep}), final)

    failed_final = os.path.join(out, 'failed.txt')
    fparts = sorted(glob.glob(failed_final + '.[0-9]*'))
    if os.path.exists(failed_final):
        fparts.insert(0, failed_final)
    failed: dict[str, str] = {}
    for part in fparts:
        with open(part, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if line:
                    failed[line.split('\t', 1)[0]] = line
    if fparts:
        tmp = failed_final + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            for utt in sorted(failed):
                f.write(failed[utt] + '\n')
        os.replace(tmp, failed_final)
        logger.info('merged %d failed files -> %s (%d utts)', len(fparts), failed_final, len(failed))
    print(f'DONE merge out={out} rows={len(keep)} utts={len({k[0] for k in keep})} '
          f'dropped_incomplete_utts={len(incomplete)} failed_utts={len(failed)}', flush=True)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def sample_candidate(llm: torch.nn.Module, inputs: dict, utt: str, k: int,
                     args: argparse.Namespace) -> tuple[list[int], bool]:
    """Draw candidate k with retries; every attempt re-seeds deterministically.

    Returns (tokens, ended_with_eos) from cosy.generate_speech_tokens_ex."""
    last_err: RuntimeError | None = None
    for attempt in range(args.max_retries + 1):
        common.seed_all(derive_seed(args.seed, utt, k, attempt))
        try:
            return cosy.generate_speech_tokens_ex(llm, inputs, temperature=args.temperature, sampling=args.sampling)
        except RuntimeError as e:  # sampling_ids max_trials (or CUDA errors) -> retry with a new seed
            last_err = e
            logger.warning('utt=%s cand=%d attempt=%d failed: %s', utt, k, attempt, str(e).splitlines()[0])
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
    assert last_err is not None
    raise last_err


def process_utt(cv, row: dict, args: argparse.Namespace, wav_root: str) -> tuple[list[dict], dict]:
    """Generate all candidates of one utterance; returns (cands.tsv rows, tokens payload); wavs written."""
    utt = row['utt']
    text_tts = row['text_tts'].strip()
    if not text_tts:
        raise ValueError('empty text_tts')
    prompt_text = row['prompt_text']  # verbatim: the manifest already applied --text_mode to it
    inputs = cosy.build_prompt_inputs(cv, text_tts, prompt_text, row['prompt_wav'], text_frontend=args.text_frontend)
    utt_dir = os.path.join(wav_root, utt)
    os.makedirs(utt_dir, exist_ok=True)
    cands: list[torch.Tensor] = []
    ended_with_eos: list[bool] = []
    rows: list[dict] = []
    for k in range(args.n_cand):
        tokens, eos_flag = sample_candidate(cv.model.llm, inputs, utt, k, args)
        tokens = [t for t in tokens if t < MAX_SPEECH_TOKEN_ID]
        if not tokens:
            raise ValueError(f'candidate {k} produced zero speech tokens')
        wav = cosy.tokens_to_wav(cv, tokens, inputs)  # [1,S] float32 cpu @ 24 kHz
        wav_path = os.path.join(utt_dir, f'{k}.wav')
        torchaudio.save(wav_path, wav, SAMPLE_RATE)
        if args.save_16k:
            torchaudio.save(os.path.join(utt_dir, f'{k}_16k.wav'), common.resample(wav, SAMPLE_RATE, 16000), 16000)
        cands.append(torch.tensor(tokens, dtype=torch.long))
        ended_with_eos.append(bool(eos_flag))
        if not eos_flag:
            logger.info('utt=%s cand=%d truncated at max_len (%d tokens, no EOS)', utt, k, len(tokens))
        rows.append({'utt': utt, 'cand': k, 'wav': os.path.abspath(wav_path),
                     'dur': f'{wav.shape[1] / SAMPLE_RATE:.3f}', 'n_tokens': len(tokens)})
    payload = {
        'utt': utt,
        'text_ids': inputs['text_ids'].detach().cpu().to(torch.long),
        'text_len_target': int(inputs['text_len_target']),
        'prompt_speech': inputs['prompt_speech'].detach().cpu().to(torch.long),
        'cands': cands,
        'ended_with_eos': ended_with_eos,
        'prompt_utt': row['prompt_utt'],
    }
    return rows, payload


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    if args.merge:
        merge_shards(args.out)
        return
    if args.manifest is None:
        raise SystemExit('--manifest is required unless --merge')
    if args.n_cand < 1:
        raise SystemExit('--n_cand must be >= 1')
    if args.temperature <= 0:
        raise SystemExit(f'--temperature must be > 0 (got {args.temperature})')
    if not 0 <= args.shard < args.nshards:
        raise SystemExit('--shard must be in [0, nshards)')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

    ckpt = resolve_ckpt(args.ckpt)  # None = stock llm.pt
    wav_root = os.path.join(args.out, 'wav')
    tok_root = os.path.join(args.out, 'tokens')
    cands_tsv, failed_txt = cands_paths(args.out, args.shard, args.nshards)

    all_rows = common.read_manifest(args.manifest)
    text_mode = derive_text_mode(all_rows)
    logger.info('manifest %s: %d rows, text_mode inferred = %s (text_tts / prompt_text used verbatim)',
                args.manifest, len(all_rows), text_mode)
    if text_mode == 'mixed':
        logger.warning('manifest text_tts does not follow a single --text_mode convention (mixed)')
    rows = common.shard(all_rows, args.shard, args.nshards)
    if args.limit > 0:
        rows = rows[:args.limit]
    logger.info('shard %d/%d -> %d rows', args.shard, args.nshards, len(rows))

    fingerprint = build_config(args, ckpt, text_mode, all_rows)
    if fingerprint['config']['ckpt_mtime'] is None:
        logger.warning('checkpoint file %s does not exist', fingerprint['config']['ckpt_file'])
    stored, write_needed = check_fingerprint(args.out, fingerprint, args.force_resume)

    # rows already in this shard's cands file (resume after a kill between the row appends and the tokens marker)
    written_keys = {_cand_key(r) for r in read_cands_rows(cands_tsv)}
    logger.info('resume state: %d rows already in %s', len(written_keys), cands_tsv)
    pending = [r for r in rows if not os.path.exists(os.path.join(tok_root, f"{r['utt']}.pt"))]
    if pending and not os.path.isfile(pending[0]['prompt_wav']):
        msg = (f"prompt_wav of the first pending row does not exist: {pending[0]['prompt_wav']} "
               f'(wrong --manifest root?)')
        if not args.dry_run:
            raise SystemExit(msg)
        logger.warning(msg)

    if args.dry_run:
        pending_utts = {r['utt'] for r in pending}
        print(f'DRY RUN fingerprint ({os.path.join(args.out, CONFIG_NAME)}, '
              f'{"matches stored" if stored and not write_needed else "would be written"}):')
        print(json.dumps(fingerprint, indent=2, ensure_ascii=False))
        print(f'DRY RUN plan: shard {args.shard}/{args.nshards} rows={len(rows)} done={len(rows) - len(pending)} '
              f'pending={len(pending)} n_cand={args.n_cand} -> {len(pending) * args.n_cand} candidates to draw')
        print('utt\tspk\tdur\tstatus')
        for r in rows:
            print(f'{r["utt"]}\t{r["spk"]}\t{float(r["dur"]):.2f}\t{"pending" if r["utt"] in pending_utts else "done"}')
        print(f'DONE dry_run shard={args.shard}/{args.nshards} rows={len(rows)} pending={len(pending)} '
              f'out={args.out} (nothing written)', flush=True)
        return

    os.makedirs(wav_root, exist_ok=True)
    os.makedirs(tok_root, exist_ok=True)
    if write_needed:
        logger.info('wrote fingerprint %s', write_fingerprint(args.out, fingerprint))

    device = torch.device(args.device)
    cv = cosy.load_cosyvoice2(device, llm_ckpt=ckpt)
    cv.model.llm.eval()
    override_sampler(cv.model.llm, args.top_p, args.top_k)
    logger.info('CosyVoice2 loaded on %s (llm ckpt=%s)', device, ckpt)

    n_done = n_skipped = n_failed = n_cands = n_dup = 0
    t0 = time.time()
    for i, row in enumerate(rows):
        utt = row['utt']
        tok_path = os.path.join(tok_root, f'{utt}.pt')
        if os.path.exists(tok_path):
            n_skipped += 1
            continue
        try:
            tsv_rows, payload = process_utt(cv, row, args, wav_root)
        except Exception as e:  # noqa: BLE001 - a failing utterance must not abort the run
            n_failed += 1
            reason = f'{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ""}'
            logger.error('utt=%s FAILED %s', utt, reason, exc_info=True)
            record_failure(failed_txt, utt, reason)
            torch.cuda.empty_cache()
            continue
        for r in tsv_rows:
            key = _cand_key(r)
            if key in written_keys:  # re-generated after a kill before the tokens marker: row already there
                n_dup += 1
                continue
            common.append_tsv(cands_tsv, r, CANDS_COLUMNS)
            written_keys.add(key)
        tmp = tok_path + '.tmp'
        torch.save(payload, tmp)
        os.replace(tmp, tok_path)  # written last: marks the utterance complete
        n_done += 1
        n_cands += len(tsv_rows)
        if n_done % 20 == 0 or i == len(rows) - 1:
            el = time.time() - t0
            logger.info('[%d/%d] done=%d skipped=%d failed=%d cands=%d elapsed=%.0fs (%.1fs/utt)',
                        i + 1, len(rows), n_done, n_skipped, n_failed, n_cands, el, el / max(1, n_done))
    logger.info('outputs: %s  %s  %s', cands_tsv, tok_root, failed_txt)
    if args.nshards > 1:
        logger.info('run `--out %s --merge` once all shards finished to build the final cands.tsv', args.out)
    print(f'DONE shard={args.shard}/{args.nshards} rows={len(rows)} done={n_done} skipped={n_skipped} '
          f'failed={n_failed} cands_written={n_cands} rows_already_present={n_dup} out={args.out}', flush=True)
    if n_done == 0 and n_failed > 0:
        logger.error('every attempted utterance failed (%d); see %s', n_failed, failed_txt)
        sys.exit(1)


if __name__ == '__main__':
    main()
