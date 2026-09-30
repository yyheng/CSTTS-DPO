#!/usr/bin/env python
"""Stage-2 multi-critic DPO fine-tuning of the CosyVoice2 speech-token LLM (paper section 2, eq. 1).

Purpose
-------
Trains the Qwen2LM (policy) initialised from the stage-1 SFT checkpoint on the preference pairs
written by 14_build_pairs.py, against a frozen deep copy of itself (reference):

    loss = -log sigmoid( beta * [ (log pi(pos) - log pi_ref(pos)) - (log pi(neg) - log pi_ref(neg)) ] )

Sequence log-probabilities follow the inference conditioning layout through
cmi_dpo.cosy.sequence_logps (EOS term included only for candidates whose generation actually
sampled EOS: pairs.pt eos_pos / eos_neg from 14_build_pairs.py, missing = True). Options:
--length_norm divides every sequence log-prob by its number of scored positions (tokens + 1 when
the EOS term is scored); --sft_weight w adds
w * mean(-log pi(pos) / len) as a regulariser; --cache_ref (default on) pre-computes the reference
log-probs once for all pairs (sharded over ranks, then all-reduced) since the reference is frozen.

The policy is wrapped in DistributedDataParallel (find_unused_parameters=True, like train.py of
the CosyVoice repo); the reference is not. bf16 autocast (--bf16) wraps only the POLICY forward;
the DPO loss is always computed in float32. The REFERENCE log-probs (the --cache_ref table and
the on-the-fly reference forward of --no-cache_ref) are ALWAYS computed in float32 with autocast
disabled, regardless of --bf16, so the reference side never carries bf16 rounding noise. What
that means for the step-0 margin (policy == reference weights): it is exactly 0 (loss log 2 =
0.6931) only when the policy also runs in float32 (no --bf16) AND the training micro-batch has
the same padding layout as the cache batch (--batch 1, or identical batch composition; the
cache is scored in rank-strided, unshuffled batches, training in shuffled ones, and float32
kernels are not bit-exact across paddings, ~1e-5 differences). Under the production setting
(--bf16, dpo.sbatch BF16=1) the policy forward is bf16 while the reference is float32, so the
step-0 margin is bf16 rounding noise (|margin| of the order of beta x 1e-1..1e0 nats on
100-700-token sequences, loss ~0.69 +- small), NOT 0. To make the real discrepancy visible, a
fresh run (no --resume, --cache_ref) pushes its first training micro-batch through the policy
under the training autocast right after the cache is built and logs
`step-0 check: max|logp_policy - logp_ref_cache| = ...` on rank 0. Gradient accumulation,
gradient clipping and --max_steps (smoke tests) are supported. Rank 0 logs loss / mean reward
margin / reward accuracy every --log_every optimizer steps (metrics are averaged over all ranks)
and also appends them to <exp>/metrics.jsonl.

Initialisation: --init_ckpt defaults to the configured stage-1 SFT checkpoint (CMI_DPO_SFT_CKPT
from cmi_dpo.paths / config/paths.env; the stock llm.pt when it is unset). --init_ckpt none
(case-insensitive) or an empty string starts from the STOCK CosyVoice2 llm.pt instead
(cosy.load_llm_only(device, llm_ckpt=None): stock llm.pt loaded with strict=False, no further
checkpoint). The frozen reference is deep-copied from the policy right after that load, i.e.
it is always the --init_ckpt model (stock or SFT). --show_paths prints the configured paths and
exits.

Pair filtering (--max_pair_tokens, default 750): pairs whose pos or neg candidate has more speech
tokens than this are skipped at load time (logged count; <= 0 disables). Candidates are
right-padded to the longest one in the micro-batch, so one very long negative in an old pairs.pt
(built before 14_build_pairs.py filtered by length) would otherwise blow up the memory of the whole
batch. Pairs with an empty pos / neg are skipped too (sequence_logps rejects them).

Model selection (--val_frac, default 0.05): a seeded (--seed) fraction of the loaded pairs is held
out and NEVER trained on: n_val = max(1, round(val_frac * n_pairs)) when val_frac > 0 and
n_pairs >= 20, else 0 (no validation set for tiny / smoke runs or --val_frac 0). Rank 0 writes the
held-out pair utts to <exp>/val_pairs.txt. After every epoch the mean DPO loss / reward margin /
reward accuracy on the held-out pairs are computed under torch.no_grad (sharded over ranks,
all-reduced; policy forward under the same --bf16 autocast as training, reference from the
float32 cache or a float32 forward), logged next to the epoch training metrics and appended to
metrics.jsonl (val_loss / val_margin / val_acc). dpo_best.pth is the epoch with the lowest
VALIDATION loss when a validation set exists, else (as before) the lowest epoch-mean training loss.

Checkpoints (rank 0, Qwen2LM state_dict WITHOUT the 'module.' prefix, loadable by
cmi_dpo.cosy.load_llm_ckpt exactly like the SFT checkpoint):
  <exp>/dpo_epoch_XXX.pth   after every epoch (or after the partial epoch cut by --max_steps)
  <exp>/dpo_best.pth        the epoch with the lowest validation loss (or, without a validation
                            set, the lowest epoch-mean training DPO loss)
  <exp>/dpo_step_XXXXXX.pth every --save_every optimizer steps (0 = off)
Alongside every dpo_epoch/dpo_step checkpoint rank 0 (over)writes <exp>/train_state.pt
{ckpt, optimizer, epoch, micro, global_step, best_loss, epoch_end_done, geometry} (best_loss =
the selection metric: validation loss when a validation set exists, else epoch training loss;
epoch_end_done = False for a dpo_step save, True for a dpo_epoch save; geometry = {world_size,
batch, accum, seed, val_frac, max_pair_tokens, n_pairs} of the run that wrote it) and, once,
<exp>/ref_cache.pt (the cached float32 reference log-probs of ALL loaded pairs, train + held-out,
indexed by their position after --max_pair_tokens filtering). Resuming a pre-empted run:
--resume <exp>/dpo_step_XXXXXX.pth (or dpo_epoch_XXX.pth) loads those weights into the POLICY
only (the reference stays the --init_ckpt model), restores the optimizer / counters from
train_state.pt when its `ckpt` entry names that checkpoint (otherwise weights-only, fresh
counters, with a warning), reuses ref_cache.pt when it matches --init_ckpt / --max_pair_tokens /
the pair count, skips the already-consumed epochs and micro-batches (the DistributedSampler order
is deterministic given --seed) and appends to metrics.jsonl. Without --resume, metrics.jsonl is
truncated at start.
Geometry check on resume: `micro` counts DataLoader micro-batches of ONE (world_size, --batch)
partition and the held-out split depends on --seed / --val_frac / --max_pair_tokens / the pair
count, so train_state.pt is refused (SystemExit) when seed, val_frac, max_pair_tokens or n_pairs
differ from the run that wrote it (held-out pairs could otherwise be trained on), and when
world_size, --batch or --accum differ while the checkpoint is mid-epoch (micro > 0: the replay
would skip the wrong micro-batches). Another GPU count / batch size IS accepted from an
epoch-boundary checkpoint (dpo_epoch_XXX.pth written after a completed epoch, micro == 0). A
train_state.pt without a geometry record (older runs) only warns.
Epoch end after a resume: a dpo_step checkpoint written on the last optimizer step of an epoch
records micro == number of micro-batches with epoch_end_done False; resuming from it runs the
pending epoch-end validation, writes dpo_epoch_XXX.pth (+ dpo_best.pth when the validation loss
improves) and then continues with the next epoch, instead of skipping that epoch's validation and
checkpoint. Without a validation set the epoch's training loss is unknown after such a resume, so
dpo_best.pth is left unchanged for that epoch (selection recorded as 'none').

Device: cuda:<LOCAL_RANK> when CUDA is available (NCCL backend), otherwise CPU (gloo backend,
DDP without device_ids) so that CPU-only smoke tests / syntax runs work; --bf16 then means CPU
bfloat16 autocast. Training runs are GPU jobs.

Environment: conda env `cosyvoicenew`. GPU job via sbatch/torchrun (<= 6 GPUs on the authors' cluster).

Example
-------
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
torchrun --nproc_per_node=4 --master_port=29517 \
    scripts/15_train_dpo.py \
    --pairs exp/round1/pairs/pairs.pt \
    --exp exp/round1/dpo --bf16 --epochs 2 --batch 4 --lr 1e-6
Smoke test (1 GPU, 5 optimizer steps):
torchrun --nproc_per_node=1 scripts/15_train_dpo.py --pairs exp/round1/pairs/pairs.pt --exp exp/dpo_smoke --max_steps 5 --log_every 1
Start from the stock CosyVoice2 LLM instead of the SFT model:
torchrun ... 15_train_dpo.py --init_ckpt none --pairs ... --exp ...
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmi_dpo import common  # noqa: E402
from cmi_dpo import cosy  # noqa: E402
from cmi_dpo import paths  # noqa: E402


class ShowPathsAction(argparse.Action):
    """--show_paths: print cmi_dpo.paths.describe() and exit, before the required flags are checked."""

    def __init__(self, option_strings: list[str], dest: str, **kwargs: Any) -> None:
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS,
                         help=kwargs.get('help', 'print the configured paths (cmi_dpo.paths) and exit'))

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
                 values: Any, option_string: str | None = None) -> None:
        print(paths.describe())
        parser.exit()


def resolve_init_ckpt(arg: str | None) -> str | None:
    """--init_ckpt value -> LLM checkpoint path, or None for the stock llm.pt.

    Flag omitted (None) = the configured CMI_DPO_SFT_CKPT (None when it is unset / empty);
    'none' (case-insensitive) or '' = the stock CosyVoice2 llm.pt; anything else = that path.
    Resolved after parsing so that --help works without config/paths.env.
    """
    if arg is None:
        return paths.sft_ckpt() or None
    return None if arg.strip().lower() in ('', 'none') else arg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split('Environment:')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--pairs', required=True, help='pairs.pt written by 14_build_pairs.py')
    p.add_argument('--exp', required=True, help='experiment directory for checkpoints, args.json and metrics.jsonl')
    p.add_argument('--init_ckpt', default=None,
                   help='LLM state_dict to initialise policy and reference from (default: the configured stage-1 SFT '
                        'checkpoint CMI_DPO_SFT_CKPT, stock llm.pt when unset); '
                        '"none" (case-insensitive) or "" = start from the stock CosyVoice2 llm.pt')
    p.add_argument('--resume', default=None,
                   help='dpo_epoch_XXX.pth / dpo_step_XXXXXX.pth of an interrupted run in --exp: loaded into the '
                        'policy only; optimizer and counters come from <exp>/train_state.pt when it matches')
    p.add_argument('--beta', type=float, default=0.1, help='DPO temperature beta')
    p.add_argument('--lr', type=float, default=1e-6, help='AdamW learning rate (constant)')
    p.add_argument('--weight_decay', type=float, default=0.0, help='AdamW weight decay')
    p.add_argument('--batch', type=int, default=4, help='pairs per GPU per micro-batch')
    p.add_argument('--accum', type=int, default=1, help='gradient accumulation steps (micro-batches per optimizer step)')
    p.add_argument('--epochs', type=int, default=2, help='number of passes over the pairs')
    p.add_argument('--grad_clip', type=float, default=1.0, help='max gradient norm (<= 0 disables clipping)')
    p.add_argument('--bf16', action='store_true', help='run the model forward under torch.autocast(bfloat16)')
    p.add_argument('--sft_weight', type=float, default=0.0,
                   help='weight w of the SFT regulariser w * mean(-log pi(pos) / len) (0 = off)')
    p.add_argument('--length_norm', action='store_true',
                   help='divide every sequence log-prob by its number of scored positions (tokens + 1 if EOS scored)')
    p.add_argument('--cache_ref', action=argparse.BooleanOptionalAction, default=True,
                   help='pre-compute the reference log-probs for all pairs once at start (always float32)')
    p.add_argument('--val_frac', type=float, default=0.05,
                   help='seeded fraction of pairs held out (never trained on) for model selection; '
                        'min 1 pair when > 0 and >= 20 pairs, 0 held out when < 20 pairs or 0')
    p.add_argument('--max_pair_tokens', type=int, default=750,
                   help='skip pairs whose pos or neg candidate has more speech tokens than this (<= 0 = off)')
    p.add_argument('--log_every', type=int, default=10, help='log metrics every N optimizer steps')
    p.add_argument('--save_every', type=int, default=0, help='save dpo_step_XXXXXX.pth every N optimizer steps (0 = off)')
    p.add_argument('--max_steps', type=int, default=0, help='stop after N optimizer steps (0 = run all epochs)')
    p.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    p.add_argument('--seed', type=int, default=0, help='random seed (shuffling, torch)')
    p.add_argument('--show_paths', action=ShowPathsAction)
    return p.parse_args()


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def _n_tokens(x) -> int:
    return int(torch.as_tensor(x).reshape(-1).numel())


def filter_pairs(pairs: list[dict], max_pair_tokens: int) -> tuple[list[dict], dict[str, int]]:
    """Drop pairs whose pos / neg candidate exceeds ``max_pair_tokens`` speech tokens (<= 0 = no
    limit) or is empty. Returns (kept pairs in the original order, {'too_long': n, 'empty': n})."""
    kept: list[dict] = []
    dropped = {'too_long': 0, 'empty': 0}
    for p in pairs:
        n_pos, n_neg = _n_tokens(p['pos']), _n_tokens(p['neg'])
        if n_pos < 1 or n_neg < 1:
            dropped['empty'] += 1
        elif max_pair_tokens > 0 and max(n_pos, n_neg) > max_pair_tokens:
            dropped['too_long'] += 1
        else:
            kept.append(p)
    return kept, dropped


def split_val(n_pairs: int, val_frac: float, seed: int) -> list[int]:
    """Sorted global indices of the held-out pairs: max(1, round(val_frac * n)) when val_frac > 0
    and n >= 20 (never more than n - 1), else [] (no validation set)."""
    if val_frac <= 0 or n_pairs < 20:
        return []
    n_val = min(n_pairs - 1, max(1, int(round(val_frac * n_pairs))))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_pairs), n_val))


class PairDataset(Dataset):
    """pairs.pt entries as LongTensors plus their GLOBAL index (position in the filtered pair list,
    used to look up cached reference log-probs). ``indices`` selects a subset (train / held-out)."""

    def __init__(self, pairs: list[dict], indices: Optional[list[int]] = None):
        self.pairs = pairs
        self.indices = list(range(len(pairs))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        g = self.indices[i]
        p = self.pairs[g]
        return {
            'idx': g,
            'utt': p['utt'],
            'text_ids': torch.as_tensor(p['text_ids']).reshape(-1).long(),
            'prompt_speech': torch.as_tensor(p['prompt_speech']).reshape(-1).long(),
            'pos': torch.as_tensor(p['pos']).reshape(-1).long(),
            'neg': torch.as_tensor(p['neg']).reshape(-1).long(),
            # EOS sampled at generation? (pairs.pt from before the flag existed -> True, old behaviour)
            'pos_eos': bool(p.get('eos_pos', True)),
            'neg_eos': bool(p.get('eos_neg', True)),
        }


def collate_pairs(items: list[dict]) -> dict:
    """Right-pad text_ids / prompt_speech / pos / neg with 0 and return their lengths (+ EOS flags)."""
    out: dict = {'idx': torch.tensor([it['idx'] for it in items], dtype=torch.long),
                 'utt': [it['utt'] for it in items]}
    for key, len_key in (('text_ids', 'text_lens'), ('prompt_speech', 'prompt_lens'),
                         ('pos', 'pos_lens'), ('neg', 'neg_lens')):
        seqs = [it[key] for it in items]
        out[key] = pad_sequence(seqs, batch_first=True, padding_value=0)
        out[len_key] = torch.tensor([s.numel() for s in seqs], dtype=torch.long)
    out['pos_eos'] = torch.tensor([it['pos_eos'] for it in items], dtype=torch.bool)
    out['neg_eos'] = torch.tensor([it['neg_eos'] for it in items], dtype=torch.bool)
    return out


def stack_pos_neg(batch: dict, device: torch.device) -> dict:
    """Stack the preferred and rejected candidates into one [2B, ...] batch (pos rows first).

    One forward per micro-batch keeps DDP's single-forward/single-backward requirement.
    """
    b = batch['pos'].size(0)
    t = max(batch['pos'].size(1), batch['neg'].size(1))
    cand = torch.zeros(2 * b, t, dtype=torch.long)
    cand[:b, :batch['pos'].size(1)] = batch['pos']
    cand[b:, :batch['neg'].size(1)] = batch['neg']
    return {
        'text_ids': torch.cat([batch['text_ids'], batch['text_ids']], dim=0).to(device),
        'text_lens': torch.cat([batch['text_lens'], batch['text_lens']], dim=0).to(device),
        'prompt_speech': torch.cat([batch['prompt_speech'], batch['prompt_speech']], dim=0).to(device),
        'prompt_lens': torch.cat([batch['prompt_lens'], batch['prompt_lens']], dim=0).to(device),
        'cand': cand.to(device),
        'cand_lens': torch.cat([batch['pos_lens'], batch['neg_lens']], dim=0).to(device),
        'eos_mask': torch.cat([batch['pos_eos'], batch['neg_eos']], dim=0).to(device),
    }


# ----------------------------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------------------------
class SeqLogpModel(nn.Module):
    """Wraps a Qwen2LM so that DDP's forward hook covers the sequence log-prob computation.

    Calling cosy.sequence_logps on the bare Qwen2LM submodules would bypass
    DistributedDataParallel.forward and hence the gradient synchronisation.
    """

    def __init__(self, llm: nn.Module):
        super().__init__()
        self.llm = llm

    def forward(self, stacked: dict) -> torch.Tensor:
        return cosy.sequence_logps(
            self.llm,
            stacked['text_ids'], stacked['text_lens'],
            stacked['prompt_speech'], stacked['prompt_lens'],
            stacked['cand'], stacked['cand_lens'],
            include_eos=True, eos_mask=stacked['eos_mask'],
        )


def init_distributed() -> tuple[int, int, int, bool, torch.device]:
    """(rank, world_size, local_rank, is_distributed, device); distributed iff launched by torchrun.

    device = cuda:<local_rank> (NCCL) when CUDA is available, else cpu (gloo; CPU smoke tests).
    """
    use_cuda = torch.cuda.is_available()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group('nccl' if use_cuda else 'gloo')
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        distributed = True
    else:
        rank, world, local_rank, distributed = 0, 1, 0, False
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    return rank, world, local_rank, distributed, device


def no_autocast(device: torch.device) -> torch.autocast:
    """Autocast DISABLED for ``device`` (float32 forward whatever --bf16 says): used for every
    reference forward so that the reference side never carries bf16 rounding noise (the step-0
    margin is then exactly 0 only when the policy runs in float32 with the same batch layout;
    see the module docstring)."""
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=False)


def all_reduce_sum(t: torch.Tensor, distributed: bool) -> torch.Tensor:
    if distributed:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


@torch.no_grad()
def precompute_ref(
    ref: SeqLogpModel,
    dataset: PairDataset,
    batch_size: int,
    device: torch.device,
    rank: int,
    world: int,
    distributed: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference sum log-probs for every pair: (lp_pos [N], lp_neg [N]) float32 on device.

    ALWAYS computed in float32 (autocast disabled, independent of --bf16): a bf16 forward depends
    strongly on the padding / batch composition, which differs between this cache (rank-strided,
    unshuffled batches) and the shuffled training micro-batches; float32 keeps that layout effect
    at ~1e-5. The policy side may still be bf16 (--bf16), so the step-0 margin is exactly 0 only
    for a float32 policy with a matching layout (module docstring).
    Each rank scores the pairs i = rank, rank+world, ...; the per-index sums are then all-reduced so
    that every rank holds the full table (every index is written by exactly one rank).
    """
    n = len(dataset)
    lp_pos = torch.zeros(n, dtype=torch.float64, device=device)
    lp_neg = torch.zeros(n, dtype=torch.float64, device=device)
    mine = list(range(rank, n, world))
    t0 = time.time()
    logging.info('computing the reference log-prob cache in float32 (autocast disabled) for %d pairs', n)
    for start in range(0, len(mine), batch_size):
        items = [dataset[i] for i in mine[start:start + batch_size]]
        batch = collate_pairs(items)
        stacked = stack_pos_neg(batch, device)
        with no_autocast(device):
            lp = ref(stacked)
        b = len(items)
        idx = batch['idx'].to(device)
        lp_pos[idx] = lp[:b].double()
        lp_neg[idx] = lp[b:].double()
        if rank == 0 and (start // batch_size) % 50 == 0:
            logging.info('ref cache: %d/%d pairs on rank 0 (%.0f s)', start + b, len(mine), time.time() - t0)
    all_reduce_sum(lp_pos, distributed)
    all_reduce_sum(lp_neg, distributed)
    return lp_pos.float(), lp_neg.float()


def dpo_terms(
    lp_pos: torch.Tensor, lp_neg: torch.Tensor,
    lr_pos: torch.Tensor, lr_neg: torch.Tensor,
    pos_lens: torch.Tensor, neg_lens: torch.Tensor,
    beta: float, length_norm: bool, sft_weight: float,
    pos_eos: torch.Tensor | None = None, neg_eos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(loss scalar, reward margin [B]) in float32.

    margin = beta * ((lp_pos - lr_pos) - (lp_neg - lr_neg)); loss = -logsigmoid(margin).mean()
    (+ sft_weight * mean(-lp_pos_raw / len_pos)). len = tokens + 1 when the EOS term was scored
    (pos_eos / neg_eos bool [B]; None = always), else tokens.
    """
    n_pos = (pos_lens + (pos_eos.long() if pos_eos is not None else 1)).float()  # scored positions
    n_neg = (neg_lens + (neg_eos.long() if neg_eos is not None else 1)).float()
    lp_pos_raw = lp_pos.float()
    lp_pos, lp_neg, lr_pos, lr_neg = (x.float() for x in (lp_pos, lp_neg, lr_pos, lr_neg))
    if length_norm:
        lp_pos, lr_pos = lp_pos / n_pos, lr_pos / n_pos
        lp_neg, lr_neg = lp_neg / n_neg, lr_neg / n_neg
    margin = beta * ((lp_pos - lr_pos) - (lp_neg - lr_neg))
    loss = -F.logsigmoid(margin).mean()
    if sft_weight > 0:
        loss = loss + sft_weight * (-lp_pos_raw / n_pos).mean()
    return loss, margin.detach()


@torch.no_grad()
def evaluate_val(
    policy: nn.Module,
    ref: SeqLogpModel,
    val_dataset: PairDataset,
    ref_cache: Optional[tuple[torch.Tensor, torch.Tensor]],
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
    world: int,
    distributed: bool,
) -> tuple[float, float, float]:
    """(mean DPO loss, mean reward margin, reward accuracy) over the held-out pairs.

    Sharded over ranks (pairs i = rank, rank+world, ...) and all-reduced. The policy forward runs
    under the same --bf16 autocast as training (bare SeqLogpModel, no DDP hooks), the reference
    comes from the float32 cache or a float32 forward. Restores policy.train() afterwards.
    """
    inner = policy.module if isinstance(policy, DDP) else policy
    inner.eval()
    tot = torch.zeros(4, dtype=torch.float64, device=device)  # [loss * B, margin sum, correct, B]
    mine = list(range(rank, len(val_dataset), world))
    for start in range(0, len(mine), args.batch):
        items = [val_dataset[i] for i in mine[start:start + args.batch]]
        batch = collate_pairs(items)
        b = len(items)
        stacked = stack_pos_neg(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16):
            lp = inner(stacked)
        lp_pos, lp_neg = lp[:b].float(), lp[b:].float()
        if ref_cache is not None:
            idx = batch['idx'].to(device)
            lr_pos, lr_neg = ref_cache[0][idx], ref_cache[1][idx]
        else:
            with no_autocast(device):
                lr = ref(stacked)
            lr_pos, lr_neg = lr[:b].float(), lr[b:].float()
        loss, margin = dpo_terms(lp_pos, lp_neg, lr_pos, lr_neg,
                                 stacked['cand_lens'][:b], stacked['cand_lens'][b:],
                                 args.beta, args.length_norm, args.sft_weight,
                                 stacked['eos_mask'][:b], stacked['eos_mask'][b:])
        tot += torch.stack([loss.detach().double() * b, margin.double().sum(), (margin > 0).double().sum(),
                            torch.tensor(float(b), dtype=torch.float64, device=device)])
    all_reduce_sum(tot, distributed)
    inner.train()
    n = max(tot[3].item(), 1.0)
    return (tot[0] / n).item(), (tot[1] / n).item(), (tot[2] / n).item()


def unwrap_llm(policy: nn.Module) -> nn.Module:
    """The bare Qwen2LM inside (DDP ->) SeqLogpModel."""
    inner = policy.module if isinstance(policy, DDP) else policy
    return inner.llm


def save_ckpt(policy: nn.Module, path: str) -> None:
    sd = {k: v.detach().cpu() for k, v in unwrap_llm(policy).state_dict().items()}
    torch.save(sd, path)
    logging.info('saved %s', path)


SPLIT_KEYS = ('seed', 'val_frac', 'max_pair_tokens', 'n_pairs')   # define the held-out split: never resumable across
LAYOUT_KEYS = ('world_size', 'batch', 'accum')                    # define the micro-batch partition: epoch boundary only


def run_geometry(args: argparse.Namespace, world: int, n_pairs: int) -> dict:
    """The run parameters a resume replay depends on (stored in train_state.pt as 'geometry')."""
    return {'world_size': int(world), 'batch': int(args.batch), 'accum': int(args.accum), 'seed': int(args.seed),
            'val_frac': float(args.val_frac), 'max_pair_tokens': int(args.max_pair_tokens), 'n_pairs': int(n_pairs)}


def save_train_state(exp: str, ckpt_path: str, optimizer: torch.optim.Optimizer,
                     epoch: int, micro: int, global_step: int, best_loss: float,
                     geometry: dict, epoch_end_done: bool) -> None:
    """Resume sidecar of the checkpoint just written: (epoch, micro) = where to continue.

    `epoch` is the 0-based index of the epoch to (re)enter and `micro` the number of its
    micro-batches already consumed (0 after a completed epoch). `epoch_end_done` is False for a
    --save_every step save (the epoch-end validation / dpo_epoch save of `epoch` has not happened
    yet, even when micro == number of micro-batches) and True for an epoch-end save. `geometry`
    (run_geometry) lets the resume refuse a replay under another partition / split. One file per
    --exp, overwritten at every save (the optimizer moments are as large as the model).
    """
    path = os.path.join(exp, 'train_state.pt')
    torch.save({'ckpt': os.path.basename(ckpt_path), 'optimizer': optimizer.state_dict(),
                'epoch': epoch, 'micro': micro, 'global_step': global_step, 'best_loss': best_loss,
                'epoch_end_done': bool(epoch_end_done), 'geometry': dict(geometry)},
               path + '.tmp')
    os.replace(path + '.tmp', path)
    logging.info('saved %s (ckpt=%s epoch=%d micro=%d step=%d epoch_end_done=%s)', path, os.path.basename(ckpt_path),
                 epoch, micro, global_step, epoch_end_done)


def check_geometry(state: dict, geometry: dict, path: str) -> None:
    """Refuse a train_state.pt whose recorded geometry makes the resume replay wrong.

    SPLIT_KEYS mismatch -> SystemExit always (the held-out split would change, held-out pairs could
    be trained on); LAYOUT_KEYS mismatch -> SystemExit while the checkpoint is mid-epoch (micro > 0),
    accepted with a warning at an epoch boundary (micro == 0: the next epoch starts a fresh
    partition). A state without a geometry record (older runs) only warns.
    """
    stored = state.get('geometry')
    if not isinstance(stored, dict):
        logging.warning('%s has no geometry record (older run): cannot verify world_size / batch / accum / seed / '
                        'val_frac / max_pair_tokens / n_pairs against the current run', path)
        return
    bad_split = [k for k in SPLIT_KEYS if stored.get(k) != geometry[k]]
    bad_layout = [k for k in LAYOUT_KEYS if stored.get(k) != geometry[k]]
    fmt = lambda keys: ', '.join(f'{k}: saved={stored.get(k)!r} now={geometry[k]!r}' for k in keys)  # noqa: E731
    if bad_split:
        raise SystemExit(f'{path} was written under another pair split ({fmt(bad_split)}); the held-out set would '
                         f'change, so this run cannot be resumed with these arguments (use the original ones, '
                         f'or start a new --exp)')
    if bad_layout and int(state.get('micro', 0)) > 0:
        raise SystemExit(f'{path} is a mid-epoch checkpoint (micro={state.get("micro")}) written under another '
                         f'micro-batch partition ({fmt(bad_layout)}); the replay would skip the wrong micro-batches. '
                         f'Resume with the original world_size / --batch / --accum, or from an epoch-boundary '
                         f'checkpoint (dpo_epoch_XXX.pth of a completed epoch)')
    if bad_layout:
        logging.warning('%s: %s changed at an epoch boundary; the next epoch starts a fresh micro-batch partition',
                        path, fmt(bad_layout))


def load_train_state(exp: str, resume_ckpt: str, geometry: dict) -> Optional[dict]:
    """train_state.pt of --exp when it belongs to `resume_ckpt` and its geometry allows the resume
    (check_geometry); None (with a warning) when the file is absent."""
    path = os.path.join(exp, 'train_state.pt')
    if not os.path.isfile(path):
        logging.warning('%s not found: resuming weights only from %s (fresh optimizer, counters from 0)', path, resume_ckpt)
        return None
    state = torch.load(path, map_location='cpu')
    if state.get('ckpt') != os.path.basename(resume_ckpt):
        raise SystemExit(f'{path} belongs to {state.get("ckpt")}, not to --resume {resume_ckpt}; '
                         f'resume from that checkpoint or delete train_state.pt for a weights-only resume')
    check_geometry(state, geometry, path)
    return state


@torch.no_grad()
def log_step0_check(policy: nn.Module, loader: DataLoader, ref_cache: tuple[torch.Tensor, torch.Tensor],
                    device: torch.device, bf16: bool) -> None:
    """Rank 0, fresh runs only: score the first training micro-batch with the (still untrained)
    policy under the training autocast and log max |logp_policy - logp_ref_cache|, i.e. the real
    step-0 policy/reference discrepancy (0 for a float32 policy with the cache's batch layout,
    bf16 rounding noise under --bf16). No collectives: the bare SeqLogpModel is used."""
    inner = policy.module if isinstance(policy, DDP) else policy
    inner.eval()
    batch = next(iter(loader))
    b = batch['pos'].size(0)
    stacked = stack_pos_neg(batch, device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
        lp = inner(stacked).float()
    idx = batch['idx'].to(device)
    diff = torch.cat([lp[:b] - ref_cache[0][idx], lp[b:] - ref_cache[1][idx]]).abs()
    logging.info('step-0 check (first micro-batch, %d pairs, policy %s vs float32 reference cache): '
                 'max|logp_policy - logp_ref_cache| = %.3e, mean %.3e (exactly 0 only for a float32 policy with the '
                 "cache's batch layout; bf16 rounding noise otherwise)", b, 'bf16' if bf16 else 'float32',
                 diff.max().item(), diff.mean().item())
    inner.train()


# ----------------------------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    args.init_ckpt = resolve_init_ckpt(args.init_ckpt)  # None = stock CosyVoice2 llm.pt (no extra checkpoint)
    rank, world, local_rank, distributed, device = init_distributed()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f'%(asctime)s %(levelname)s [rank {rank}] %(message)s',
    )
    if device.type != 'cuda':
        logging.warning('CUDA not available: running on CPU (smoke / syntax runs only)')
    common.seed_all(args.seed)
    if rank == 0:
        os.makedirs(args.exp, exist_ok=True)
        with open(os.path.join(args.exp, 'args.json'), 'w', encoding='utf-8') as f:
            json.dump({**vars(args), 'world_size': world}, f, indent=2)
    logging.info('world=%d device=%s args=%s', world, device, vars(args))

    pairs_all = torch.load(args.pairs, map_location='cpu')
    if len(pairs_all) == 0:
        raise SystemExit(f'no pairs in {args.pairs}')
    pairs, dropped = filter_pairs(pairs_all, args.max_pair_tokens)
    logging.info('%d pairs loaded from %s: %d skipped (pos or neg > --max_pair_tokens %d), %d skipped (empty), %d kept',
                 len(pairs_all), args.pairs, dropped['too_long'], args.max_pair_tokens, dropped['empty'], len(pairs))
    if len(pairs) == 0:
        raise SystemExit(f'no pairs left in {args.pairs} after --max_pair_tokens {args.max_pair_tokens} filtering')
    val_idx = split_val(len(pairs), args.val_frac, args.seed)
    val_set = set(val_idx)
    train_idx = [i for i in range(len(pairs)) if i not in val_set]
    dataset = PairDataset(pairs)                   # all kept pairs (reference cache index space)
    train_dataset = PairDataset(pairs, train_idx)  # never contains a held-out pair
    val_dataset = PairDataset(pairs, val_idx) if val_idx else None
    if rank == 0:
        with open(os.path.join(args.exp, 'val_pairs.txt'), 'w', encoding='utf-8') as f:
            for i in val_idx:
                f.write(f"{pairs[i]['utt']}\n")
    sampler = DistributedSampler(train_dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed,
                                 drop_last=False)
    loader = DataLoader(train_dataset, batch_size=args.batch, sampler=sampler, collate_fn=collate_pairs,
                        num_workers=args.num_workers, drop_last=False)
    logging.info('%d train pairs / %d held-out pairs (--val_frac %.3f, seed %d; utts in %s), %d micro-batches per '
                 'epoch per rank; dpo_best.pth chosen by %s', len(train_dataset), len(val_idx), args.val_frac,
                 args.seed, os.path.join(args.exp, 'val_pairs.txt'), len(loader),
                 'validation loss' if val_dataset is not None else 'epoch training loss (no validation set)')

    policy_llm = cosy.load_llm_only(device, llm_ckpt=args.init_ckpt)  # None = stock llm.pt only
    policy_llm.train()
    common.seed_all(args.seed)  # the CosyVoice yaml resets the global RNGs to 1986 while loading
    ref_llm = copy.deepcopy(policy_llm).eval()  # the reference is ALWAYS the --init_ckpt model
    for p in ref_llm.parameters():
        p.requires_grad_(False)
    ref = SeqLogpModel(ref_llm).eval()
    if args.resume:
        cosy.load_llm_ckpt(policy_llm, args.resume)  # policy only, after the reference copy
    policy: nn.Module = SeqLogpModel(policy_llm)
    if distributed:
        if device.type == 'cuda':
            policy = DDP(policy, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        else:
            policy = DDP(policy, find_unused_parameters=True)
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    init_desc = args.init_ckpt or f'stock CosyVoice2 llm.pt ({paths.cosy_model_dir()})'
    logging.info('policy initialised from %s (%.1f M trainable params); reference frozen copy of %s',
                 args.resume or init_desc, n_train / 1e6, init_desc)

    ref_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    ref_cache_path = os.path.join(args.exp, 'ref_cache.pt')
    if args.cache_ref:
        if args.resume and os.path.isfile(ref_cache_path):
            saved = torch.load(ref_cache_path, map_location='cpu')
            if (saved.get('init_ckpt') == args.init_ckpt and saved.get('n') == len(dataset)
                    and saved.get('max_pair_tokens', args.max_pair_tokens) == args.max_pair_tokens):
                ref_cache = (saved['lp_pos'].to(device), saved['lp_neg'].to(device))
                logging.info('reference log-probs for %d pairs reloaded from %s', len(dataset), ref_cache_path)
            else:
                logging.warning('%s does not match --init_ckpt / --max_pair_tokens / pair count; recomputing',
                                ref_cache_path)
        if ref_cache is None:
            t0 = time.time()
            ref_cache = precompute_ref(ref, dataset, args.batch, device, rank, world, distributed)
            logging.info('reference log-probs cached in float32 for %d pairs (train + held-out) in %.0f s '
                         '(mean lp_pos %.2f, lp_neg %.2f)', len(dataset), time.time() - t0,
                         ref_cache[0].mean().item(), ref_cache[1].mean().item())
            if rank == 0:
                torch.save({'init_ckpt': args.init_ckpt, 'pairs': os.path.abspath(args.pairs), 'n': len(dataset),
                            'max_pair_tokens': args.max_pair_tokens, 'dtype': 'float32',
                            'lp_pos': ref_cache[0].cpu(), 'lp_neg': ref_cache[1].cpu()}, ref_cache_path)
    else:
        logging.info('--no-cache_ref: the reference is scored on the fly in float32 (autocast disabled)')
    if ref_cache is not None and not args.resume and rank == 0 and len(train_dataset) > 0:
        log_step0_check(policy, loader, ref_cache, device, args.bf16)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    def autocast() -> torch.autocast:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16)

    metrics_path = os.path.join(args.exp, 'metrics.jsonl')
    geometry = run_geometry(args, world, len(dataset))

    global_step = 0
    best_loss = float('inf')
    start_epoch = 0
    skip_micro = 0  # micro-batches of start_epoch already consumed before the resume point
    resumed_epoch_end_pending = False  # step save at the end of start_epoch whose epoch-end block never ran
    if args.resume:
        state = load_train_state(args.exp, args.resume, geometry)
        if state is not None:
            optimizer.load_state_dict(state['optimizer'])
            global_step, best_loss = int(state['global_step']), float(state['best_loss'])
            start_epoch, skip_micro = int(state['epoch']), int(state['micro'])
            resumed_epoch_end_pending = not bool(state.get('epoch_end_done', True))  # old sidecars: assume done
            logging.info('resumed optimizer + counters: step %d, epoch %d, %d micro-batches consumed, best loss %.4f%s',
                         global_step, start_epoch + 1, skip_micro, best_loss,
                         ' (epoch-end block of that epoch still pending)' if resumed_epoch_end_pending else '')
        if rank == 0:
            with open(metrics_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'resumed_from': args.resume, 'epoch': start_epoch + 1, 'step': global_step,
                                    'micro': skip_micro, 'time': time.strftime('%Y-%m-%dT%H:%M:%S')}) + '\n')
    elif rank == 0:
        open(metrics_path, 'w', encoding='utf-8').close()  # fresh run: never interleave with an older run's records
    stop = False
    t_start = time.time()
    # running sums for logging: [loss * B, margin sum, correct count, B]
    run = torch.zeros(4, dtype=torch.float64, device=device)
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        policy.train()
        ep = torch.zeros(4, dtype=torch.float64, device=device)
        optimizer.zero_grad(set_to_none=True)
        n_micro = len(loader)
        micro_done = 0
        for i, batch in enumerate(loader):
            if epoch == start_epoch and i < skip_micro:
                continue  # deterministic sampler order: replay past the resume point without a forward
            micro_done = i + 1
            b = batch['pos'].size(0)
            is_step = ((i + 1) % args.accum == 0) or (i + 1 == n_micro)
            # micro-batches in this optimizer step's group (the trailing group of an epoch may be shorter)
            n_in_group = min(args.accum, n_micro - (i // args.accum) * args.accum)
            stacked = stack_pos_neg(batch, device)
            sync_ctx = policy.no_sync() if (distributed and not is_step) else nullcontext()
            with sync_ctx:
                with autocast():
                    lp = policy(stacked)
                lp_pos, lp_neg = lp[:b].float(), lp[b:].float()
                if ref_cache is not None:
                    idx = batch['idx'].to(device)
                    lr_pos, lr_neg = ref_cache[0][idx], ref_cache[1][idx]
                else:
                    with torch.no_grad(), no_autocast(device):  # reference always float32
                        lr = ref(stacked)
                    lr_pos, lr_neg = lr[:b].float(), lr[b:].float()
                loss, margin = dpo_terms(lp_pos, lp_neg, lr_pos, lr_neg,
                                         stacked['cand_lens'][:b], stacked['cand_lens'][b:],
                                         args.beta, args.length_norm, args.sft_weight,
                                         stacked['eos_mask'][:b], stacked['eos_mask'][b:])
                (loss / n_in_group).backward()
            stats = torch.stack([loss.detach().double() * b, margin.double().sum(),
                                 (margin > 0).double().sum(), torch.tensor(float(b), dtype=torch.float64, device=device)])
            run += stats
            ep += stats
            if not is_step:
                continue
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            else:
                grad_norm = torch.tensor(0.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % args.log_every == 0:
                tot = all_reduce_sum(run.clone(), distributed)
                run.zero_()
                if rank == 0:
                    rec = {
                        'epoch': epoch + 1, 'step': global_step, 'micro': i + 1,
                        'loss': (tot[0] / tot[3]).item(), 'margin': (tot[1] / tot[3]).item(),
                        'acc': (tot[2] / tot[3]).item(), 'grad_norm': float(grad_norm),
                        'lr': optimizer.param_groups[0]['lr'], 'elapsed_s': time.time() - t_start,
                    }
                    logging.info('epoch %d step %d (%d/%d): loss %.4f margin %.4f acc %.3f grad_norm %.3f lr %.2e (%.0f s)',
                                 rec['epoch'], rec['step'], rec['micro'], n_micro, rec['loss'], rec['margin'],
                                 rec['acc'], rec['grad_norm'], rec['lr'], rec['elapsed_s'])
                    with open(metrics_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(rec) + '\n')
            if args.save_every > 0 and global_step % args.save_every == 0 and rank == 0:
                step_path = os.path.join(args.exp, f'dpo_step_{global_step:06d}.pth')
                save_ckpt(policy, step_path)
                # epoch_end_done=False even when micro_done == n_micro: the validation / dpo_epoch save below
                # has not happened yet, and a resume from this file must still run it
                save_train_state(args.exp, step_path, optimizer, epoch, micro_done, global_step, best_loss,
                                 geometry, epoch_end_done=False)
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop = True
                break
        tot = all_reduce_sum(ep.clone(), distributed)
        trained = tot[3].item() > 0
        # every micro-batch of this epoch was consumed before the resume point and the epoch-end block never
        # ran (step save on the last optimizer step of the epoch): run validation / checkpoint now
        epoch_end_pending = (not trained and epoch == start_epoch and resumed_epoch_end_pending
                             and skip_micro >= n_micro)
        if not trained and not epoch_end_pending:  # resumed past the end of an epoch already cut by --max_steps
            logging.info('epoch %d: nothing left to train after the resume point', epoch + 1)
            continue
        if trained:
            ep_loss, ep_margin, ep_acc = (tot[0] / tot[3]).item(), (tot[1] / tot[3]).item(), (tot[2] / tot[3]).item()
        else:
            ep_loss = ep_margin = ep_acc = float('nan')  # training metrics of the epoch are unknown after the resume
            logging.info('epoch %d: all %d micro-batches were consumed before the resume point; running the pending '
                         'epoch-end validation / checkpoint', epoch + 1, n_micro)
        val_rec: dict = {}
        if val_dataset is not None:  # all ranks take part (sharded + all-reduced)
            t_val = time.time()
            val_loss, val_margin, val_acc = evaluate_val(policy, ref, val_dataset, ref_cache, args, device,
                                                         rank, world, distributed)
            val_rec = {'val_loss': val_loss, 'val_margin': val_margin, 'val_acc': val_acc, 'n_val': len(val_dataset)}
            sel_loss, sel_name = val_loss, 'val loss'
            if rank == 0:
                logging.info('==== epoch %d validation on %d held-out pairs: loss %.4f margin %.4f acc %.3f (%.0f s) ====',
                             epoch + 1, len(val_dataset), val_loss, val_margin, val_acc, time.time() - t_val)
        elif trained:
            sel_loss, sel_name = ep_loss, 'train loss'
        else:  # no validation set and no training metrics: dpo_best.pth cannot be judged for this epoch
            sel_loss, sel_name = float('inf'), 'none'
        if rank == 0:
            logging.info('==== epoch %d done: mean train loss %.4f margin %.4f acc %.3f (%d steps, %.2f h) ====',
                         epoch + 1, ep_loss, ep_margin, ep_acc, global_step, (time.time() - t_start) / 3600)
            ep_path = os.path.join(args.exp, f'dpo_epoch_{epoch + 1:03d}.pth')
            save_ckpt(policy, ep_path)
            if sel_loss < best_loss:
                best_loss = sel_loss
                shutil.copyfile(ep_path, os.path.join(args.exp, 'dpo_best.pth'))
                logging.info('new best epoch %d (%s %.4f) -> dpo_best.pth', epoch + 1, sel_name, sel_loss)
            elif sel_name == 'none':
                logging.warning('epoch %d: no validation set and no training metrics after the resume; dpo_best.pth '
                                'left unchanged', epoch + 1)
            # a --max_steps cut leaves the epoch unfinished: resume re-enters it after micro_done batches
            if stop:
                save_train_state(args.exp, ep_path, optimizer, epoch, micro_done, global_step, best_loss,
                                 geometry, epoch_end_done=True)
            else:
                save_train_state(args.exp, ep_path, optimizer, epoch + 1, 0, global_step, best_loss,
                                 geometry, epoch_end_done=True)
            with open(metrics_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'epoch': epoch + 1, 'step': global_step,
                                    'epoch_loss': ep_loss if trained else None,
                                    'epoch_margin': ep_margin if trained else None,
                                    'epoch_acc': ep_acc if trained else None, **val_rec,
                                    'best_loss': best_loss if best_loss != float('inf') else None,  # strict JSON
                                    'selection': sel_name}) + '\n')
        if distributed:
            dist.barrier()
        if stop:
            logging.info('stopping after --max_steps %d', args.max_steps)
            break
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
