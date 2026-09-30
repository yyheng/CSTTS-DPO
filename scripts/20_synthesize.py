#!/usr/bin/env python3
"""Synthesise a code-switching augmentation corpus with a fine-tuned CosyVoice2 (see docs/DESIGN.md).

One utterance per manifest row (train transcripts + prompt assignment from 00_prep_manifest.py),
generated with the same conditioning path as 10_gen_candidates.py (frontend inputs built once per
row, RAS sampling at --temperature, flow + HiFT vocoding), written as a Kaldi-style directory:

    <out>/wav/<utt>.wav      mono, --sr Hz (default 16000)
    <out>/wav.scp            "<utt> <abs wav>"          <out>/text     "<utt> <text_raw>"
    <out>/utt2spk            "<utt> <spk>"              <out>/utt2dur  "<utt> <seconds>"
    <out>/failed.txt         "<utt>\\t<reason>"         (generation errors AND rejected utterances)
    <out>/synth_config.json  config fingerprint (see below)

Text: text_tts and prompt_text are taken from the manifest VERBATIM. 00_prep_manifest.py applied
--text_mode (strip | raw) to both columns, so nothing is re-stripped here; the mode is inferred
from the manifest (rows whose text_raw carries <...> tags) and recorded in the fingerprint.

Quality filter (Whisper fine-tuning cannot take clips > 30 s): an utterance is REJECTED when its
generation was truncated by max_len without sampling EOS (cosy.generate_speech_tokens_ex ->
ended_with_eos False) or when its vocoded wav is longer than --max_dur (default 30 s). A rejected
draw is re-sampled with a fresh seed (--reject_retries, default 1 = retry once, same seeded
attempt counter as the sampler-error retries); if every draw is rejected the utt is written to
failed.txt as "<utt>\\trejected: <reason>" and NO wav / Kaldi line is written. --keep_truncated
disables the EOS filter, --max_dur inf disables the duration filter.
On a resume an utt whose LAST line in this shard's failed file is a "rejected: ..." line counts as
done (skipped, counted as rejected_skipped in the DONE line): the draws are deterministic, so
re-drawing it would only repeat the same rejections and append duplicate lines. --retry_rejected
opts back in (useful together with a larger --reject_retries, --keep_truncated or --max_dur, all of
which change the fingerprint and therefore need --force_resume). Utts whose last line is another
failure (sampler error, OOM, ...) are retried on every resume as before.

Row order: the manifest is in sorted utt (= speaker) order, so a --hours cap on the raw order would
drop the alphabetically last speakers. The manifest rows are therefore SHUFFLED with
random.Random(--seed) BEFORE sharding, --limit and the cap, so every shard and the capped subset are
speaker-balanced and reproducible (--no_shuffle keeps the manifest order). The shard files are thus
in shuffled order; the merge / finalize pass sorts the final wav.scp/text/utt2spk/utt2dur by utt.

Config fingerprint: <out>/synth_config.json stores the checkpoint (path, mtime, size), temperature,
sampling, top_p/top_k (null = model default), seed, text_frontend, text_mode, max_dur,
keep_truncated, shuffle, sr, utt_prefix and manifest_sha256 (common.manifest_content_hash: an
order-independent hash of the utt / text_raw / text_tts / prompt_utt / prompt_wav / prompt_text
columns, so a manifest re-generated at the same path with another prompt assignment, --seed or
--text_mode is detected; the manifest path / row count are informational). On a resume (existing
outputs) the current config is compared with the stored one and the run ABORTS on any difference
unless --force_resume is given (then the differences are logged and the fingerprint is overwritten).
A stored fingerprint that lacks a newer key (e.g. manifest_sha256) cannot be checked for it: it only
warns and the fingerprint is re-written. Outputs that predate the fingerprint (no synth_config.json)
only produce a warning and get one written.

The four Kaldi files are appended line by line after each wav is saved, so a killed job resumes
where it stopped: an utt counts as done only when it is listed in ALL FOUR of this shard's files
and its wav is on disk (a record torn by a kill between the four appends is regenerated
deterministically; the duplicate lines are removed by the merge/finalize step below).
Sharding: with --nshards > 1 each shard appends to <out>/wav.scp.<shard> (and text/utt2spk/utt2dur
and failed.txt likewise) to avoid interleaved appends over NFS; run `--merge` once afterwards (no
GPU) to concatenate + sort + de-duplicate the shard files into <out>/wav.scp etc., keeping only utts
present in all four files (dropped utts are reported), and to merge <out>/failed.txt.<shard> (plus an
existing <out>/failed.txt) into <out>/failed.txt, one line per utt (the last one wins), utts that
made it into the corpus removed. With --nshards 1 the plain files are written directly and the same
sort/de-duplicate/consistency pass runs at the end of the job.
--hours caps the cumulative synthesised duration of the CORPUS (docs/DESIGN.md): each shard
stops at hours / nshards (common.shard is round-robin over the shuffled rows, so shards are
duration- and speaker-balanced); durations already in the shard's utt2dur are counted on resume.
E.g. 100 h over 6 shards -> --hours 100 on every shard.
Deterministic: each utterance re-seeds all RNGs from sha256(seed, utt, attempt), independent of
sharding and of the shuffle.
--dry_run builds the fingerprint and the plan (shuffled/sharded/limited rows with their
done/pending status and the estimated cap cut-off) WITHOUT loading the model, prints both and exits
0; a fingerprint mismatch aborts the dry run exactly like a real run would.
Exit status is 1 when every attempted utterance failed or was rejected (so downstream steps that
require exit status 0 stop).

Environment: one GPU, the CosyVoice env (torch 2.3.1+cu121),
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1.
Paths: the CosyVoice2 checkout / model dir and the default --ckpt (CMI_DPO_SFT_CKPT) come from
cmi_dpo.paths (the CMI_DPO_* environment variables); --show_paths prints them and exits.

Example:
    python -u scripts/20_synthesize.py \
        --manifest data/train.tsv \
        --ckpt exp/dpo_round1/dpo_best.pth \
        --out exp/synth_dpo1 --hours 100 --shard 0 --nshards 6
    python -u scripts/20_synthesize.py --out exp/synth_dpo1 --merge
"""
from __future__ import annotations

import argparse
import functools
import glob
import hashlib
import json
import logging
import os
import random
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

KALDI_FILES = ('wav.scp', 'text', 'utt2spk', 'utt2dur')
MODEL_SR = 24000
MAX_SPEECH_TOKEN_ID = 6561  # EOS id; the vocoder never sees it
CONFIG_NAME = 'synth_config.json'
CONFIG_KEYS = ('ckpt', 'ckpt_file', 'ckpt_mtime', 'ckpt_size', 'temperature', 'sampling', 'top_p', 'top_k',
               'seed', 'text_frontend', 'text_mode', 'max_dur', 'keep_truncated', 'shuffle', 'sr', 'utt_prefix',
               'manifest_sha256')

logger = logging.getLogger('synthesize')


class UtteranceRejected(Exception):
    """Raised when every draw of an utterance failed the quality filter (truncated / too long)."""


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
        description='Synthesise one utterance per manifest row into a Kaldi-style corpus (SFT or DPO CosyVoice2 LLM).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--manifest', default=None, help='manifest TSV from 00_prep_manifest.py (required unless --merge)')
    p.add_argument('--out', required=True, help='output corpus dir')
    p.add_argument('--ckpt', default=None,
                   help='Qwen2LM state_dict: stage-1 SFT or DPO checkpoint (same loader); default = the configured '
                        'CMI_DPO_SFT_CKPT (stock llm.pt when unset); "none" = stock llm.pt')
    p.add_argument('--sr', type=int, default=16000, help='output wav sample rate (model runs at 24000)')
    p.add_argument('--hours', type=float, default=0.0,
                   help='corpus-level cap: stop when the cumulative synthesised duration >= hours (0 = no cap); '
                        'each shard stops at hours / nshards')
    p.add_argument('--shard', type=int, default=0, help='this shard index (0-based)')
    p.add_argument('--nshards', type=int, default=1, help='number of shards the manifest is split into')
    p.add_argument('--temperature', type=float, default=1.0, help='softmax temperature on LLM logits')
    p.add_argument('--sampling', type=int, default=25, help='`sampling` argument of Qwen2LM.sampling_ids')
    p.add_argument('--top_p', type=float, default=None, help='override RAS nucleus top_p (model default 0.8)')
    p.add_argument('--top_k', type=int, default=None, help='override RAS nucleus top_k (model default 25)')
    p.add_argument('--seed', type=int, default=0,
                   help='base seed: seeds the manifest shuffle and the per-utterance sampling seeds')
    p.add_argument('--limit', type=int, default=0, help='process at most this many rows of the shard (0 = all)')
    p.add_argument('--text_frontend', action='store_true',
                   help='apply frontend.text_normalize (ttsfrd) before tokenising; default keeps SEAME text as-is')
    p.add_argument('--max_retries', type=int, default=2,
                   help='extra sampling attempts (fresh seed) when the sampler hits its max_trials error')
    p.add_argument('--max_dur', type=float, default=30.0,
                   help='reject utterances whose vocoded wav is longer than this many seconds '
                        '(Whisper fine-tuning takes <= 30 s clips); "inf" disables the filter')
    p.add_argument('--keep_truncated', action='store_true',
                   help='keep utterances whose generation hit max_len without EOS (default: reject them)')
    p.add_argument('--reject_retries', type=int, default=1,
                   help='re-sample a rejected (truncated / too long) utterance this many times with a fresh seed '
                        'before writing it to failed.txt')
    p.add_argument('--retry_rejected', action='store_true',
                   help='on resume, re-draw utterances already listed as "rejected: ..." in this shard\'s failed '
                        'file (default: they count as done, since the draws are deterministic)')
    p.add_argument('--no_shuffle', action='store_true',
                   help='keep the manifest (sorted utt) order instead of the seeded shuffle before sharding/capping')
    p.add_argument('--utt_prefix', default='', help='prefix added to output utt ids (e.g. "syn_" to avoid clashes with real train)')
    p.add_argument('--force_resume', action='store_true',
                   help='resume even when <out>/synth_config.json differs from the current config (fingerprint is overwritten)')
    p.add_argument('--dry_run', action='store_true',
                   help='print the fingerprint + the planned rows (no model load, nothing written) and exit 0')
    p.add_argument('--merge', action='store_true',
                   help='no synthesis: concatenate + sort + de-duplicate <out>/{wav.scp,text,utt2spk,utt2dur}.<shard> '
                        'into the final files (utts missing from any of the four files are dropped)')
    p.add_argument('--device', default='cuda:0', help='torch device')
    p.add_argument('--show_paths', action=ShowPathsAction)
    return p.parse_args()


def derive_seed(base: int, utt: str, attempt: int) -> int:
    """Stable 31-bit seed for one utterance under base seed."""
    h = hashlib.sha256(f'{base}|{utt}|{attempt}'.encode('utf-8')).digest()
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
        'sampling': int(args.sampling),
        'top_p': args.top_p,
        'top_k': args.top_k,
        'seed': int(args.seed),
        'text_frontend': bool(args.text_frontend),
        'text_mode': text_mode,
        'max_dur': float(args.max_dur),
        'keep_truncated': bool(args.keep_truncated),
        'shuffle': not args.no_shuffle,
        'sr': int(args.sr),
        'utt_prefix': args.utt_prefix,
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
    """True when the corpus dir already holds synthesis outputs (Kaldi files, shard files or wavs)."""
    if not os.path.isdir(out):
        return False
    for name in KALDI_FILES:
        if glob.glob(os.path.join(out, name)) or glob.glob(os.path.join(out, name + '.[0-9]*')):
            return True
    wav_root = os.path.join(out, 'wav')
    return os.path.isdir(wav_root) and bool(os.listdir(wav_root))


def check_fingerprint(out: str, current: dict, force_resume: bool) -> tuple[dict | None, bool]:
    """Compare the current fingerprint with <out>/synth_config.json.

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
            raise SystemExit(msg + '\nRefusing to mix outputs of different configs in one corpus dir; use a new '
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
# Kaldi IO
# ---------------------------------------------------------------------------
def kaldi_paths(out: str, shard: int, nshards: int) -> dict[str, str]:
    suffix = '' if nshards == 1 else f'.{shard}'
    return {name: os.path.join(out, name + suffix) for name in KALDI_FILES}


def read_kaldi_lines(path: str) -> dict[str, str]:
    """utt -> full line of one Kaldi file (last occurrence wins; missing file -> empty)."""
    lines: dict[str, str] = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if line:
                    lines[line.split(maxsplit=1)[0]] = line
    return lines


def read_existing(kpaths: dict[str, str]) -> tuple[set[str], float]:
    """Utts already COMPLETE in this shard (listed in all four Kaldi files) and their seconds from utt2dur.

    A record torn by a kill between the four appends is absent from at least one file, so it is not
    'done' and gets regenerated (deterministically); the merge/finalize pass de-duplicates the lines."""
    tables = {name: read_kaldi_lines(kpaths[name]) for name in KALDI_FILES}
    done = set.intersection(*(set(t) for t in tables.values()))
    partial = set().union(*(set(t) for t in tables.values())) - done
    if partial:
        logger.warning('%d utts are listed in only some of the shard files (torn record) and will be regenerated: %s%s',
                       len(partial), ' '.join(sorted(partial)[:10]), ' ...' if len(partial) > 10 else '')
    seconds = 0.0
    for utt in done:
        parts = tables['utt2dur'][utt].split()
        if len(parts) == 2:
            seconds += float(parts[1])
    return done, seconds


def append_line(path: str, line: str) -> None:
    with open(path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def read_failed_lines(path: str) -> dict[str, str]:
    """utt -> LAST "<utt>\\t<reason>" line of a failed file (missing file -> empty)."""
    lines: dict[str, str] = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if line:
                    lines[line.split('\t', 1)[0]] = line
    return lines


def read_rejected(failed_path: str) -> set[str]:
    """Output utts whose last failed-file line is a quality rejection ("<utt>\\trejected: ...")."""
    return {utt for utt, line in read_failed_lines(failed_path).items()
            if line.split('\t', 1)[1:] and line.split('\t', 1)[1].startswith('rejected:')}


def merge_failed(out: str, corpus_utts: set[str]) -> int:
    """Merge <out>/failed.txt.<shard> (+ an existing <out>/failed.txt) into <out>/failed.txt: one line per
    utt (the last occurrence wins), sorted, utts present in the merged corpus removed (a regenerated
    torn record or a --retry_rejected success supersedes its old failure). Returns the utt count."""
    final = os.path.join(out, 'failed.txt')
    parts = sorted(glob.glob(final + '.[0-9]*'))
    if os.path.exists(final):
        parts.insert(0, final)
    if not parts:
        return 0
    failed: dict[str, str] = {}
    for part in parts:
        failed.update(read_failed_lines(part))
    superseded = [u for u in failed if u in corpus_utts]
    for u in superseded:
        del failed[u]
    tmp = final + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        for utt in sorted(failed):
            f.write(failed[utt] + '\n')
    os.replace(tmp, final)
    logger.info('merged %d failed file(s) -> %s (%d utts; %d superseded by corpus entries)', len(parts), final,
                len(failed), len(superseded))
    return len(failed)


def merge_shards(out: str, require_shards: bool = True) -> None:
    """Build the final <out>/{wav.scp,text,utt2spk,utt2dur} from <out>/<name>.<shard> files plus any
    existing final file: sorted by utt (the shard files are in shuffled order), de-duplicated by utt
    (last occurrence wins), restricted to utts present in all four files (partially written records
    are dropped and reported), written atomically. <out>/failed.txt.<shard> files are merged into
    <out>/failed.txt likewise (merge_failed).
    With require_shards=False (the --nshards 1 finalize pass) the plain files alone are accepted."""
    tables: dict[str, dict[str, str]] = {}
    n_parts = 0
    for name in KALDI_FILES:
        final = os.path.join(out, name)
        parts = sorted(glob.glob(final + '.[0-9]*'))
        if require_shards and not parts:
            raise SystemExit(f'--merge: no shard files {name}.<shard> found in {out}')
        if os.path.exists(final):
            parts.insert(0, final)
        n_parts += len(parts)
        lines: dict[str, str] = {}
        for part in parts:
            lines.update(read_kaldi_lines(part))
        tables[name] = lines
    if n_parts == 0:
        logger.info('merge: no Kaldi files in %s, nothing to do', out)
        return
    keep = set.intersection(*(set(t) for t in tables.values()))
    dropped = set().union(*(set(t) for t in tables.values())) - keep
    if dropped:
        logger.warning('dropping %d utts missing from at least one of %s: %s%s', len(dropped), '/'.join(KALDI_FILES),
                       ' '.join(sorted(dropped)[:10]), ' ...' if len(dropped) > 10 else '')
    for name in KALDI_FILES:
        final = os.path.join(out, name)
        tmp = final + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            for utt in sorted(keep):
                f.write(tables[name][utt] + '\n')
        os.replace(tmp, final)
        logger.info('wrote %s (%d lines, sorted by utt)', final, len(keep))
    n_failed = merge_failed(out, keep)
    print(f'DONE merge out={out} utts={len(keep)} dropped_incomplete={len(dropped)} failed_utts={n_failed}', flush=True)


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------
def synthesize_utt(cv, row: dict, args: argparse.Namespace) -> tuple[torch.Tensor, dict]:
    """Build inputs, sample tokens (seeded retries) and vocode; returns ([1,S] float32 cpu @ 24 kHz, info).

    Attempt a re-seeds all RNGs from sha256(seed, utt, a); a is advanced both by sampler errors
    (budget --max_retries) and by quality rejections (budget --reject_retries: truncated generation
    without EOS unless --keep_truncated, or wav longer than --max_dur). Raises UtteranceRejected
    when the last allowed draw is still rejected; the rejected draw is never returned."""
    utt = row['utt']
    text_tts = row['text_tts'].strip()
    if not text_tts:
        raise ValueError('empty text_tts')
    prompt_text = row['prompt_text']  # verbatim: the manifest already applied --text_mode to it
    inputs = cosy.build_prompt_inputs(cv, text_tts, prompt_text, row['prompt_wav'], text_frontend=args.text_frontend)
    attempt = n_err = n_rej = 0
    while True:
        common.seed_all(derive_seed(args.seed, utt, attempt))
        try:
            tokens, ended_with_eos = cosy.generate_speech_tokens_ex(cv.model.llm, inputs, temperature=args.temperature,
                                                                    sampling=args.sampling)
        except RuntimeError as e:  # sampling_ids max_trials (or CUDA errors) -> retry with a new seed
            logger.warning('utt=%s attempt=%d failed: %s', utt, attempt, str(e).splitlines()[0])
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
            n_err += 1
            if n_err > args.max_retries:
                raise
            attempt += 1
            continue
        tokens = [t for t in tokens if t < MAX_SPEECH_TOKEN_ID]
        if not tokens:
            raise ValueError('generation produced zero speech tokens')
        reason = None
        wav = None
        if not ended_with_eos and not args.keep_truncated:
            reason = f'truncated at max_len ({len(tokens)} tokens, no EOS)'
        else:
            wav = cosy.tokens_to_wav(cv, tokens, inputs)
            dur = wav.shape[1] / MODEL_SR
            if dur > args.max_dur:
                reason = f'too long ({dur:.2f} s > --max_dur {args.max_dur:g} s, {len(tokens)} tokens)'
        if reason is None:
            assert wav is not None
            return wav, {'attempt': attempt, 'n_tokens': len(tokens), 'ended_with_eos': bool(ended_with_eos),
                         'n_rejected_draws': n_rej}
        n_rej += 1
        if n_rej > args.reject_retries:
            raise UtteranceRejected(f'{reason} after {n_rej} draw(s)')
        logger.info('utt=%s attempt=%d rejected (%s): re-sampling with a fresh seed', utt, attempt, reason)
        attempt += 1


def plan_rows(rows: list[dict], done: set[str], rejected: set[str], seconds: float, cap_seconds: float,
              utt_prefix: str) -> list[dict]:
    """Dry-run plan: status (done / rejected / pending / beyond_cap) of every row in processing order
    plus the estimated cap cut-off.

    The cap estimate uses the manifest (ground-truth) duration of pending rows as a proxy for the
    synthesised duration, so it is approximate."""
    plan: list[dict] = []
    est = seconds
    for r in rows:
        out_utt = utt_prefix + r['utt']
        if out_utt in done:
            status = 'done'
        elif out_utt in rejected:
            status = 'rejected'
        elif cap_seconds > 0 and est >= cap_seconds:
            status = 'beyond_cap'
        else:
            status = 'pending'
            est += float(r['dur'])
        plan.append({'utt': out_utt, 'spk': r['spk'], 'dur': float(r['dur']), 'status': status})
    return plan


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    if args.merge:
        merge_shards(args.out)
        return
    if args.manifest is None:
        raise SystemExit('--manifest is required unless --merge')
    if not 0 <= args.shard < args.nshards:
        raise SystemExit('--shard must be in [0, nshards)')
    if args.temperature <= 0:
        raise SystemExit(f'--temperature must be > 0 (got {args.temperature})')
    if args.hours < 0:
        raise SystemExit('--hours must be >= 0')
    if not args.max_dur > 0:
        raise SystemExit(f'--max_dur must be > 0 (got {args.max_dur}); use inf to disable the filter')
    if args.reject_retries < 0 or args.max_retries < 0:
        raise SystemExit('--reject_retries / --max_retries must be >= 0')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

    ckpt = resolve_ckpt(args.ckpt)  # None = stock llm.pt
    wav_root = os.path.join(args.out, 'wav')
    kpaths = kaldi_paths(args.out, args.shard, args.nshards)
    failed_txt = os.path.join(args.out, 'failed.txt' + ('' if args.nshards == 1 else f'.{args.shard}'))
    done, seconds = read_existing(kpaths)
    cap_seconds = args.hours * 3600.0 / args.nshards  # corpus-level cap split evenly over the round-robin shards
    logger.info('resume state: %d utts / %.2f h already complete in %s', len(done), seconds / 3600.0, kpaths['wav.scp'])
    rejected = read_rejected(failed_txt) - done  # a utt that later made it into the corpus is not rejected
    if rejected and args.retry_rejected:
        logger.info('%d utts listed as rejected in %s will be re-drawn (--retry_rejected)', len(rejected), failed_txt)
        rejected = set()
    elif rejected:
        logger.info('%d utts listed as rejected in %s are skipped (deterministic draws; --retry_rejected re-draws them)',
                    len(rejected), failed_txt)

    all_rows = common.read_manifest(args.manifest)
    text_mode = derive_text_mode(all_rows)
    logger.info('manifest %s: %d rows, text_mode inferred = %s (text_tts / prompt_text used verbatim)',
                args.manifest, len(all_rows), text_mode)
    if text_mode == 'mixed':
        logger.warning('manifest text_tts does not follow a single --text_mode convention (mixed)')
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(all_rows)  # speaker-balanced shards / cap; independent of the sampling seeds
    rows = common.shard(all_rows, args.shard, args.nshards)
    if args.limit > 0:
        rows = rows[:args.limit]
    logger.info('shard %d/%d -> %d rows (%s order); cap=%.2f h corpus -> %.2f h for this shard', args.shard, args.nshards,
                len(rows), 'manifest' if args.no_shuffle else f'seeded shuffle, seed {args.seed}', args.hours,
                cap_seconds / 3600.0)

    fingerprint = build_config(args, ckpt, text_mode, all_rows)
    if fingerprint['config']['ckpt_mtime'] is None:
        logger.warning('checkpoint file %s does not exist', fingerprint['config']['ckpt_file'])
    stored, write_needed = check_fingerprint(args.out, fingerprint, args.force_resume)

    pending = [r for r in rows if args.utt_prefix + r['utt'] not in rejected
               and not (args.utt_prefix + r['utt'] in done
                        and os.path.exists(os.path.join(wav_root, f"{args.utt_prefix}{r['utt']}.wav")))]
    if pending and not os.path.isfile(pending[0]['prompt_wav']):
        msg = (f"prompt_wav of the first pending row does not exist: {pending[0]['prompt_wav']} "
               f'(wrong --manifest root?)')
        if not args.dry_run:
            raise SystemExit(msg)
        logger.warning(msg)

    if args.dry_run:
        plan = plan_rows(rows, done, rejected, seconds, cap_seconds, args.utt_prefix)
        counts = {s: sum(1 for p in plan if p['status'] == s) for s in ('done', 'rejected', 'pending', 'beyond_cap')}
        print(f'DRY RUN fingerprint ({os.path.join(args.out, CONFIG_NAME)}, '
              f'{"matches stored" if stored and not write_needed else "would be written"}):')
        print(json.dumps(fingerprint, indent=2, ensure_ascii=False))
        print(f'DRY RUN plan: shard {args.shard}/{args.nshards} rows={len(plan)} done={counts["done"]} '
              f'rejected={counts["rejected"]} pending={counts["pending"]} beyond_cap={counts["beyond_cap"]} '
              f'(est. {(seconds + sum(p["dur"] for p in plan if p["status"] == "pending")) / 3600.0:.2f} h after run; '
              f'cap {cap_seconds / 3600.0:.2f} h/shard; est. uses manifest durations)')
        print('utt\tspk\tdur\tstatus')
        for p in plan:
            print(f'{p["utt"]}\t{p["spk"]}\t{p["dur"]:.2f}\t{p["status"]}')
        print(f'DONE dry_run shard={args.shard}/{args.nshards} rows={len(plan)} pending={counts["pending"]} '
              f'out={args.out} (nothing written)', flush=True)
        return

    os.makedirs(wav_root, exist_ok=True)
    if write_needed:
        logger.info('wrote fingerprint %s', write_fingerprint(args.out, fingerprint))

    device = torch.device(args.device)
    cv = cosy.load_cosyvoice2(device, llm_ckpt=ckpt)
    cv.model.llm.eval()
    override_sampler(cv.model.llm, args.top_p, args.top_k)
    logger.info('CosyVoice2 loaded on %s (llm ckpt=%s)', device, ckpt)

    n_done = n_skipped = n_failed = n_rejected = n_redraw = n_rejected_skipped = 0
    capped = False
    t0 = time.time()
    for i, row in enumerate(rows):
        if cap_seconds > 0 and seconds >= cap_seconds:
            capped = True
            logger.info('hours cap reached for this shard: %.2f h >= %.2f h (corpus cap %.2f h / %d shards)',
                        seconds / 3600.0, cap_seconds / 3600.0, args.hours, args.nshards)
            break
        out_utt = args.utt_prefix + row['utt']
        wav_path = os.path.join(wav_root, f'{out_utt}.wav')
        if out_utt in done and os.path.exists(wav_path):
            n_skipped += 1
            continue
        if out_utt in rejected:  # rejected by an earlier run of this shard; deterministic -> not re-drawn
            n_rejected_skipped += 1
            continue
        try:
            wav, info = synthesize_utt(cv, row, args)
        except UtteranceRejected as e:  # quality filter: no wav, no Kaldi lines
            n_rejected += 1
            logger.warning('utt=%s REJECTED %s', row['utt'], e)
            append_line(failed_txt, f'{out_utt}\trejected: {e}')
            continue
        except Exception as e:  # noqa: BLE001 - a failing utterance must not abort the run
            n_failed += 1
            reason = f'{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ""}'
            logger.error('utt=%s FAILED %s', row['utt'], reason, exc_info=True)
            append_line(failed_txt, f'{out_utt}\t{reason}')
            torch.cuda.empty_cache()
            continue
        n_redraw += info['n_rejected_draws']
        if args.sr != MODEL_SR:
            wav = common.resample(wav, MODEL_SR, args.sr)
        dur = wav.shape[1] / args.sr
        torchaudio.save(wav_path, wav, args.sr)
        append_line(kpaths['wav.scp'], f'{out_utt} {os.path.abspath(wav_path)}')
        append_line(kpaths['text'], f'{out_utt} {row["text_raw"].strip()}')
        append_line(kpaths['utt2spk'], f'{out_utt} {row["spk"]}')
        append_line(kpaths['utt2dur'], f'{out_utt} {dur:.3f}')
        done.add(out_utt)
        seconds += dur
        n_done += 1
        if n_done % 50 == 0 or i == len(rows) - 1:
            el = time.time() - t0
            logger.info('[%d/%d] done=%d skipped=%d failed=%d rejected=%d redraws=%d synth=%.2f h elapsed=%.0fs (%.1fs/utt)',
                        i + 1, len(rows), n_done, n_skipped, n_failed, n_rejected, n_redraw, seconds / 3600.0, el,
                        el / max(1, n_done))
    logger.info('kaldi files: %s', ' '.join(kpaths.values()))
    if args.nshards == 1:
        merge_shards(args.out, require_shards=False)  # sort by utt + de-duplicate + drop torn records in the plain files
    else:
        logger.info('run `--out %s --merge` once all shards finished to build the final (utt-sorted) Kaldi files', args.out)
    print(f'DONE shard={args.shard}/{args.nshards} rows={len(rows)} done={n_done} skipped={n_skipped} '
          f'failed={n_failed} rejected={n_rejected} rejected_skipped={n_rejected_skipped} redraws={n_redraw} '
          f'hours_in_shard={seconds / 3600.0:.3f} capped={capped} out={args.out}', flush=True)
    if n_done == 0 and (n_failed + n_rejected) > 0:
        logger.error('every attempted utterance failed (%d) or was rejected (%d); see %s', n_failed, n_rejected, failed_txt)
        sys.exit(1)


if __name__ == '__main__':
    main()
