#!/usr/bin/env python
"""Stage-1 SFT of the CosyVoice2 LLM (Qwen2LM) on SEAME token caches (see docs/DESIGN.md).

Mirrors the CosyVoice train.py of the original setup ($CMI_DPO_COSY_ROOT/train.py): torchrun DDP
(``dist.init_process_group``, ``LOCAL_RANK`` for the device / ``dist.get_rank()`` for the
samplers, ``DistributedSampler``, collate via ``pad_sequence(padding_value=0)``) over the LLM
only. ``Qwen2LM.forward(batch, device)`` takes ``text_token [B,L]``, ``text_token_len [B]``,
``speech_token [B,T]``, ``speech_token_len [B]`` and returns ``{'loss', 'acc'}``.

Policy is built with ``cmi_dpo.cosy.load_llm_only(device, llm_ckpt=--init_ckpt)`` (default
``none`` = the pretrained ``$CMI_DPO_COSY_MODEL_DIR/llm.pt``; the configured CMI_DPO_SFT_CKPT is
the OUTPUT of this stage, never its default input). Model paths come from cmi_dpo.paths
(the CMI_DPO_* environment variables); ``--show_paths`` prints them and exits.

Optimisation (paper section 4: AdamW, lr 2e-4, linear warm-up, "batch size of 4")
  * ``--lr_schedule constant_with_warmup`` (default): linear warm-up over ``--warmup_steps``
    optimizer steps to ``--lr``, then constant (the paper only states a linear warm-up).
    ``--lr_schedule linear``: warm-up then linear decay to 0 at the last optimizer step (the
    previous behaviour of this script). Both come from ``transformers.get_scheduler``.
    ``--warmup_steps -1`` = 10 % of the total optimizer steps (like train.py).
  * ``--batch`` is the batch size PER GPU (micro-batch). The paper's "batch size of 4" is
    reproduced per GPU here, so the effective batch is 4 x number of GPUs (x ``--accum``).
    ``--global_batch G`` instead fixes the effective batch: the gradient accumulation factor
    is set to G / (batch x world_size), which must divide exactly (error otherwise) and
    overrides ``--accum``.
  * Gradient accumulation: every micro-batch loss is divided by the number of micro-batches
    in its group and back-propagated (DDP gradient sync only on the last one); one optimizer
    step + one scheduler step per group; ``--max_steps``, ``--eval_every`` and ``--log_every``
    count OPTIMIZER steps. The trailing group of an epoch may be shorter.
  * gradient clipping, optional bf16 autocast, periodic validation with early stopping on
    validation loss, rank-0 logging (lr, train loss/acc, val loss/acc, elapsed hours).

Checkpoints (rank 0; plain ``state_dict`` without the ``module.`` prefix, loadable by
``cosy.load_llm_ckpt``): ``<exp>/epoch_XXX.pth`` after every epoch (or after the partial epoch
cut by ``--max_steps`` / early stopping) and ``<exp>/best.pth`` on every val-loss improvement.
Next to every one of them rank 0 atomically (over)writes ``<exp>/train_state.pt`` =
{ckpt, optimizer, scheduler, epoch, micro, step, best_val, best_step, bad_evals, epoch_end_done,
geometry}, the same sidecar pattern as 15_train_dpo.py (one file per --exp, it names the
checkpoint it belongs to). ``epoch_end_done`` is False for a best.pth written by a validation
and True for an epoch_XXX.pth written by the epoch-end block; ``geometry`` = {world_size, batch,
accum (resolved, i.e. after --global_batch), seed} of the run that wrote it.

Resuming: ``--resume <ckpt>`` loads those weights into the model (instead of --init_ckpt) and,
when ``<exp>/train_state.pt`` exists and its ``ckpt`` entry names that checkpoint, restores the
optimizer, the scheduler, the epoch/step counters and the best val loss, then skips the
already-consumed epochs and micro-batches (the DistributedSampler order is deterministic given
--seed). train_state.pt missing = weights-only resume (fresh optimizer, counters from 0, with a
warning); train_state.pt naming ANOTHER checkpoint = error (resume from that one or delete
the file). ``--max_steps`` / ``--epochs`` may be raised on resume: the counters continue and
the schedule is rebuilt for the new totals.
Geometry check: ``micro`` counts DataLoader micro-batches of ONE (world_size, --batch) partition
in the --seed order, so a mid-epoch checkpoint (micro > 0) is refused (SystemExit) when
world_size, --batch, the resolved accum or --seed differ from the run that wrote it (the replay
would skip the wrong micro-batches); at an epoch boundary (micro == 0) a change only warns.
A train_state.pt without a geometry record (older runs) only warns.
Pending epoch end: a best.pth whose validation fell on the last optimizer step of an epoch
records micro == number of micro-batches with epoch_end_done False; resuming from it runs the
epoch-end block of that epoch (the validation already happened at that step and is not repeated,
so ``bad_evals`` is not touched; epoch_XXX.pth and the boundary sidecar are written) even when
that step was the last one allowed by --max_steps / --epochs, then continues with the next epoch
instead of skipping that epoch's checkpoint.

Device: CUDA + NCCL when available; without CUDA the script falls back to the gloo backend on
CPU (smoke tests only; ``--bf16`` is then a no-op).

Environment: GPU, the CosyVoice env. Must be launched with torchrun.

Example (2 GPUs, from the repo root):
    torchrun --standalone --nproc_per_node=2 \\
        scripts/02_train_sft.py \\
        --train_cache 'data/sft_cache/train.shard*.pt' \\
        --valid_cache data/sft_cache/valid.pt \\
        --exp exp/sft_seame --epochs 20 --bf16 --global_batch 8
Resume a pre-empted run (same --exp):
    ... --resume exp/sft_seame/epoch_003.pth
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import logging
import math
import os
import sys
import time
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from transformers import get_scheduler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmi_dpo import common  # noqa: E402
from cmi_dpo import cosy  # noqa: E402
from cmi_dpo import paths  # noqa: E402

LOG = logging.getLogger("train_sft")
LR_SCHEDULES = ("constant_with_warmup", "linear")


class ShowPathsAction(argparse.Action):
    """--show_paths: print cmi_dpo.paths.describe() and exit, before the required flags are checked."""

    def __init__(self, option_strings: list[str], dest: str, **kwargs: Any) -> None:
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS,
                         help=kwargs.get("help", "print the configured paths (cmi_dpo.paths) and exit"))

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
                 values: Any, option_string: str | None = None) -> None:
        print(paths.describe())
        parser.exit()


# --------------------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--train_cache", required=True,
                   help="Training cache(s) from 01_build_sft_cache.py: one or more .pt paths, "
                        "comma-separated and/or glob patterns (quote globs).")
    p.add_argument("--valid_cache", required=True,
                   help="Validation cache(s), same syntax as --train_cache.")
    p.add_argument("--exp", required=True,
                   help="Experiment dir: checkpoints, train_state.pt, train.log, args.json, best.json go here.")
    p.add_argument("--lr", type=float, default=2e-4, help="Peak AdamW learning rate (paper: 2e-4).")
    p.add_argument("--lr_schedule", choices=LR_SCHEDULES, default="constant_with_warmup",
                   help="constant_with_warmup: linear warm-up then constant lr (paper); "
                        "linear: warm-up then linear decay to 0 at the last optimizer step.")
    p.add_argument("--batch", type=int, default=4,
                   help="Batch size PER GPU (micro-batch; paper: 4). Effective batch = batch x GPUs x accum.")
    p.add_argument("--accum", type=int, default=1,
                   help="Gradient accumulation factor (micro-batches per optimizer step); "
                        "overridden by --global_batch.")
    p.add_argument("--global_batch", type=int, default=None,
                   help="Effective batch size over all GPUs: sets accum = global_batch / (batch x world_size); "
                        "error if that is not an integer.")
    p.add_argument("--epochs", type=int, default=20, help="Maximum number of epochs.")
    p.add_argument("--max_steps", type=int, default=50000,
                   help="Maximum number of OPTIMIZER steps (global, all ranks step together). "
                        "Training stops at min(max_steps, epochs * steps_per_epoch).")
    p.add_argument("--warmup_steps", type=int, default=-1,
                   help="Linear warmup in optimizer steps; -1 = 10%% of the total steps (like train.py).")
    p.add_argument("--eval_every", type=int, default=1000,
                   help="Validate every N optimizer steps (also at the end of every epoch).")
    p.add_argument("--patience", type=int, default=5,
                   help="Early stopping: stop after this many consecutive validations without "
                        "improvement of val loss (0 = disabled).")
    p.add_argument("--min_delta", type=float, default=0.0,
                   help="Minimum val-loss improvement that resets the patience counter.")
    p.add_argument("--save_every_epoch", dest="save_every_epoch", action="store_true", default=True,
                   help="Save <exp>/epoch_XXX.pth after every epoch (default, like train.py and "
                        "docs/DESIGN.md; best.pth is always saved on improvement).")
    p.add_argument("--no_save_every_epoch", dest="save_every_epoch", action="store_false",
                   help="Only save epoch_XXX.pth for the last epoch.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for sampler shuffling and the global RNGs (applied after the model "
                        "is built, because loading cosyvoice.yaml resets them to 1986).")
    p.add_argument("--init_ckpt", default=None,
                   help="Optional Qwen2LM state_dict to start from (with or without 'module.' "
                        "prefix). Default / 'none' = the pretrained $CMI_DPO_COSY_MODEL_DIR/llm.pt "
                        "(the configured SFT_CKPT is this stage's output, not its input). Ignored with --resume.")
    p.add_argument("--resume", default=None,
                   help="Checkpoint written by this script (<exp>/epoch_XXX.pth or best.pth): load its "
                        "weights and, when <exp>/train_state.pt names it, the optimizer / scheduler / "
                        "counters / best val loss; already-consumed epochs are skipped.")
    p.add_argument("--grad_clip", type=float, default=5.0,
                   help="Max gradient norm for clip_grad_norm_ (<= 0 disables).")
    p.add_argument("--bf16", action="store_true", help="Run forward/backward under bf16 autocast (CUDA only).")
    p.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay.")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers per rank.")
    p.add_argument("--log_every", type=int, default=100,
                   help="Log running train loss/acc every N optimizer steps.")
    p.add_argument("--max_eval_batches", type=int, default=0,
                   help="Cap the number of validation batches per rank (0 = whole valid cache).")
    p.add_argument("--show_paths", action=ShowPathsAction)
    args = p.parse_args(argv)
    if args.init_ckpt is not None and args.init_ckpt.strip().lower() in ("", "none"):
        args.init_ckpt = None  # stock llm.pt (cosy.load_llm_only skips the extra checkpoint)
    if args.accum < 1:
        p.error(f"--accum {args.accum} must be >= 1")
    if args.global_batch is not None and args.global_batch < 1:
        p.error(f"--global_batch {args.global_batch} must be >= 1")
    if args.resume is not None and not os.path.isfile(args.resume):
        p.error(f"--resume {args.resume}: file not found")
    return args


def resolve_accum(args: argparse.Namespace, world_size: int) -> int:
    """Gradient accumulation factor: --global_batch / (batch x world_size) when given, else --accum."""
    if args.global_batch is None:
        return args.accum
    denom = args.batch * world_size
    if args.global_batch % denom != 0:
        raise SystemExit(f"--global_batch {args.global_batch} is not divisible by "
                         f"--batch {args.batch} x world_size {world_size} = {denom}")
    return args.global_batch // denom


# --------------------------------------------------------------------------- data
class TokenCacheDataset(Dataset):
    """List of {'utt', 'text_token' int32 [L], 'speech_token' int32 [T]} entries."""

    def __init__(self, entries: list[dict]) -> None:
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        e = self.entries[idx]
        return {
            "text_token": e["text_token"].to(torch.int32),
            "speech_token": e["speech_token"].to(torch.int32),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Right-pad with 0 and attach int32 lengths (Qwen2LM.forward unpads with them)."""
    text_tokens = [s["text_token"] for s in batch]
    speech_tokens = [s["speech_token"] for s in batch]
    return {
        "text_token": pad_sequence(text_tokens, batch_first=True, padding_value=0),
        "text_token_len": torch.tensor([t.shape[0] for t in text_tokens], dtype=torch.int32),
        "speech_token": pad_sequence(speech_tokens, batch_first=True, padding_value=0),
        "speech_token_len": torch.tensor([t.shape[0] for t in speech_tokens], dtype=torch.int32),
    }


def expand_cache_paths(spec: str) -> list[str]:
    """Expand a comma-separated list of paths / glob patterns into sorted existing files."""
    paths: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        hits = sorted(glob.glob(part)) if any(c in part for c in "*?[") else [part]
        if not hits:
            raise FileNotFoundError(f"no cache file matches {part!r}")
        paths.extend(hits)
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"cache file(s) not found: {missing}")
    return paths


def load_caches(spec: str) -> list[dict]:
    """Load and concatenate one or more cache shards, validating the entry schema."""
    entries: list[dict] = []
    for path in expand_cache_paths(spec):
        part = torch.load(path, map_location="cpu")
        if not isinstance(part, list):
            raise ValueError(f"{path}: expected a list of dict entries, got {type(part).__name__}")
        for e in part:
            if not {"utt", "text_token", "speech_token"} <= set(e):
                raise ValueError(f"{path}: entry missing keys, has {sorted(e)}")
        LOG.info("loaded %d entries from %s", len(part), path)
        entries.extend(part)
    if not entries:
        raise ValueError(f"no entries loaded from {spec!r}")
    return entries


# --------------------------------------------------------------------------- train / eval
def forward_prop(batch: dict, model: torch.nn.Module, device: torch.device,
                 bf16: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Qwen2LM.forward and return (loss, acc) tensors (bf16 autocast only on CUDA)."""
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=bf16 and device.type == "cuda"):
        out = model(batch, device)
    return out["loss"], out["acc"]


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device,
             bf16: bool, max_batches: int) -> tuple[float, float]:
    """Token-weighted val loss/acc over all ranks (all-reduced so every rank sees the same numbers).

    Qwen2LM.forward returns per-batch means over the non-IGNORE targets (speech tokens + one
    EOS per utterance; length_normalized_loss=True, th_accuracy), so each batch is re-weighted
    by that target count and the sums are all-reduced: partial last batches and unequal
    per-rank batch counts do not bias the result. The forward goes through the underlying
    Qwen2LM (not the DDP wrapper, whose forward broadcasts buffers = a collective) so ranks
    may run different numbers of batches.
    """
    core = model.module if isinstance(model, DDP) else model
    core.eval()
    tot = torch.zeros(3, dtype=torch.float64, device=device)  # loss_sum, correct_sum, n_targets
    for i, batch in enumerate(loader):
        if 0 < max_batches <= i:
            break
        n_tgt = int(batch["speech_token_len"].sum()) + int(batch["speech_token"].shape[0])
        loss, acc = forward_prop(to_device(batch, device), core, device, bf16)
        tot[0] += float(loss.item()) * n_tgt
        tot[1] += float(acc.item()) * n_tgt
        tot[2] += float(n_tgt)
    dist.all_reduce(tot, op=dist.ReduceOp.SUM)
    n = max(tot[2].item(), 1.0)
    core.train()
    return tot[0].item() / n, tot[1].item() / n


def unwrapped_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """state_dict of the underlying Qwen2LM (no 'module.' prefix), tensors on CPU."""
    core = model.module if isinstance(model, DDP) else model
    return {k: v.detach().cpu() for k, v in core.state_dict().items()}


def save_ckpt(model: torch.nn.Module, path: str) -> None:
    """Atomic write of the unwrapped state_dict (tmp + os.replace)."""
    torch.save(unwrapped_state_dict(model), path + ".tmp")
    os.replace(path + ".tmp", path)
    LOG.info("saved %s", path)


GEOMETRY_KEYS = ("world_size", "batch", "accum", "seed")  # define the micro-batch partition / replay order


def run_geometry(args: argparse.Namespace, world_size: int, accum: int) -> dict:
    """The run parameters the resume replay depends on (stored in train_state.pt as 'geometry')."""
    return {"world_size": int(world_size), "batch": int(args.batch), "accum": int(accum), "seed": int(args.seed)}


def save_train_state(exp: str, ckpt_path: str, optimizer: torch.optim.Optimizer,
                     scheduler: Any, epoch: int, micro: int, step: int, best_val: float,
                     best_step: int, bad_evals: int, geometry: dict, epoch_end_done: bool) -> None:
    """Resume sidecar of the checkpoint just written: (epoch, micro) = where to continue.

    ``epoch`` is the 0-based index of the epoch to (re)enter and ``micro`` the number of its
    micro-batches already consumed (0 after a completed epoch). ``epoch_end_done`` is False for
    a best.pth written by a validation (the epoch-end block of ``epoch`` has not run yet, even
    when micro == number of micro-batches) and True for an epoch-end save. ``geometry``
    (run_geometry) lets the resume refuse a mid-epoch replay under another partition. One file
    per --exp, overwritten atomically at every save (the AdamW moments are twice the model size).
    """
    path = os.path.join(exp, "train_state.pt")
    torch.save({"ckpt": os.path.basename(ckpt_path), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "micro": micro, "step": step,
                "best_val": best_val, "best_step": best_step, "bad_evals": bad_evals,
                "epoch_end_done": bool(epoch_end_done), "geometry": dict(geometry)},
               path + ".tmp")
    os.replace(path + ".tmp", path)
    LOG.info("saved %s (ckpt=%s epoch=%d micro=%d step=%d best_val=%.4f epoch_end_done=%s)", path,
             os.path.basename(ckpt_path), epoch, micro, step, best_val, epoch_end_done)


def check_geometry(state: dict, geometry: dict, path: str) -> None:
    """Refuse a mid-epoch train_state.pt (micro > 0) written under another world_size / batch /
    accum / seed (the micro-batch replay would skip the wrong batches); at an epoch boundary a
    change only warns. A state without a geometry record (older runs) only warns."""
    stored = state.get("geometry")
    if not isinstance(stored, dict):
        LOG.warning("%s has no geometry record (older run): cannot verify world_size / batch / accum / seed "
                    "against the current run", path)
        return
    bad = [k for k in GEOMETRY_KEYS if stored.get(k) != geometry[k]]
    if not bad:
        return
    desc = ", ".join(f"{k}: saved={stored.get(k)!r} now={geometry[k]!r}" for k in bad)
    if int(state.get("micro", 0)) > 0:
        raise SystemExit(f"{path} is a mid-epoch checkpoint (micro={state.get('micro')}) written under another "
                         f"micro-batch partition ({desc}); the replay would skip the wrong micro-batches. Resume "
                         f"with the original GPU count / --batch / --accum (--global_batch) / --seed, or from an "
                         f"epoch-boundary checkpoint (epoch_XXX.pth of a completed epoch)")
    LOG.warning("%s: %s changed at an epoch boundary; the next epoch starts a fresh micro-batch partition",
                path, desc)


def load_train_state(exp: str, resume_ckpt: str, geometry: dict) -> Optional[dict]:
    """train_state.pt of --exp when it belongs to ``resume_ckpt`` and its geometry allows the
    resume (check_geometry); None (with a warning) if absent."""
    path = os.path.join(exp, "train_state.pt")
    if not os.path.isfile(path):
        LOG.warning("%s not found: resuming weights only from %s (fresh optimizer, counters from 0)",
                    path, resume_ckpt)
        return None
    state = torch.load(path, map_location="cpu")
    if state.get("ckpt") != os.path.basename(resume_ckpt):
        raise SystemExit(f"{path} belongs to {state.get('ckpt')}, not to --resume {resume_ckpt}; "
                         f"resume from that checkpoint or delete train_state.pt for a weights-only resume")
    check_geometry(state, geometry, path)
    return state


def setup_logging(rank: int, exp: str) -> None:
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if rank == 0:
        os.makedirs(exp, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(exp, "train.log")))
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING,
                        format=fmt, handlers=handlers, force=True)


def build_loader(entries: list[dict], batch: int, world_size: int, rank: int, shuffle: bool,
                 seed: int, num_workers: int, pin_memory: bool) -> tuple[DataLoader, DistributedSampler]:
    """Training loader: DistributedSampler (global rank) over the whole cache."""
    ds = TokenCacheDataset(entries)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed)
    loader = DataLoader(ds, batch_size=batch, sampler=sampler, collate_fn=collate_fn,
                        num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    return loader, sampler


def build_eval_loader(entries: list[dict], batch: int, world_size: int, rank: int,
                      num_workers: int, pin_memory: bool) -> DataLoader:
    """Validation loader: this rank's strided slice, WITHOUT the DistributedSampler padding
    (which repeats utterances so that every rank gets the same count); see evaluate()."""
    ds = Subset(TokenCacheDataset(entries), list(range(rank, len(entries), world_size)))
    return DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate_fn,
                      num_workers=num_workers, pin_memory=pin_memory, drop_last=False)


def train(args: argparse.Namespace) -> None:
    use_cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if use_cuda else "gloo")
    local_rank = int(os.environ["LOCAL_RANK"])  # GPU index on this node
    rank = dist.get_rank()                       # global rank: samplers, rank-0 gating
    world_size = dist.get_world_size()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    setup_logging(rank, args.exp)
    LOG.info("rank %d/%d (local_rank %d): device %s, backend %s", rank, world_size, local_rank,
             device, dist.get_backend())
    if not use_cuda:
        LOG.warning("CUDA not available: CPU/gloo fallback (smoke tests only; --bf16 ignored)")
    if rank == 0:
        with open(os.path.join(args.exp, "args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    # model ---------------------------------------------------------------
    weights = args.resume or args.init_ckpt
    LOG.info("building Qwen2LM (%s=%s)", "resume" if args.resume else "init_ckpt", weights)
    llm = cosy.load_llm_only(device, llm_ckpt=weights)
    # cosyvoice.yaml re-seeds python/numpy/torch/cuda to 1986 on load: seed AFTER the model.
    common.seed_all(args.seed)
    llm.train()
    model = DDP(llm, device_ids=[local_rank] if use_cuda else None,
                output_device=local_rank if use_cuda else None, find_unused_parameters=True)
    n_params = sum(p.numel() for p in llm.parameters())
    LOG.info("Qwen2LM parameters: %.1f M", n_params / 1e6)

    # data ----------------------------------------------------------------
    train_entries = load_caches(args.train_cache)
    valid_entries = load_caches(args.valid_cache)
    train_loader, train_sampler = build_loader(train_entries, args.batch, world_size, rank, True,
                                               args.seed, args.num_workers, use_cuda)
    valid_loader = build_eval_loader(valid_entries, args.batch, world_size, rank, args.num_workers,
                                     use_cuda)
    accum = resolve_accum(args, world_size)
    micro_per_epoch = len(train_loader)
    steps_per_epoch = math.ceil(micro_per_epoch / accum)
    total_steps = min(args.max_steps, args.epochs * steps_per_epoch)
    warmup = args.warmup_steps if args.warmup_steps >= 0 else int(0.1 * total_steps)
    LOG.info("train %d utts, valid %d utts, batch %d per GPU x %d GPUs x accum %d = effective batch %d, "
             "%d micro-batches/epoch, %d optimizer steps/epoch, %d total steps, warmup %d, "
             "lr_schedule %s", len(train_entries), len(valid_entries), args.batch, world_size, accum,
             args.batch * world_size * accum, micro_per_epoch, steps_per_epoch, total_steps, warmup,
             args.lr_schedule)

    # optim ---------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(args.lr_schedule, optimizer=optimizer, num_warmup_steps=warmup,
                              num_training_steps=total_steps)

    # loop state (possibly restored) ---------------------------------------
    geometry = run_geometry(args, world_size, accum)
    best_val = float("inf")
    best_step = 0
    bad_evals = 0
    step = 0
    start_epoch = 0
    skip_micro = 0  # micro-batches of start_epoch already consumed before the resume point
    resumed_epoch_end_pending = False  # best.pth written on the last step of start_epoch, epoch-end block never ran
    if args.resume:
        state = load_train_state(args.exp, args.resume, geometry)
        if state is not None:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            step, best_val = int(state["step"]), float(state["best_val"])
            best_step, bad_evals = int(state.get("best_step", 0)), int(state.get("bad_evals", 0))
            start_epoch, skip_micro = int(state["epoch"]), int(state["micro"])
            resumed_epoch_end_pending = not bool(state.get("epoch_end_done", True))  # old sidecars: assume done
            LOG.info("resumed optimizer + scheduler + counters from %s: step %d, epoch %d, "
                     "%d micro-batches consumed, best val_loss %.4f @%d, bad_evals %d, lr %.8f%s",
                     args.resume, step, start_epoch + 1, skip_micro, best_val, best_step, bad_evals,
                     scheduler.get_last_lr()[0],
                     " (epoch-end block of that epoch still pending)" if resumed_epoch_end_pending else "")
        if start_epoch >= args.epochs:
            LOG.warning("resume point (epoch %d) is past --epochs %d: nothing to train", start_epoch + 1,
                        args.epochs)
        if step >= total_steps:
            LOG.warning("resumed step %d >= total steps %d: nothing to train (raise --max_steps/--epochs)",
                        step, total_steps)
    stop = False
    start = time.time()
    run_loss, run_acc, run_n = 0.0, 0.0, 0

    def elapsed_h() -> float:
        return (time.time() - start) / 3600.0

    def run_eval(epoch: int, micro_done: int) -> None:
        nonlocal best_val, best_step, bad_evals, stop
        val_loss, val_acc = evaluate(model, valid_loader, device, args.bf16, args.max_eval_batches)
        lr = scheduler.get_last_lr()[0]
        LOG.info("[eval] epoch %d step %d/%d: lr = %.8f, val_loss = %.4f, val_acc = %.4f, "
                 "best = %.4f @%d, time = %.3f h", epoch + 1, step, total_steps, lr, val_loss,
                 val_acc, best_val, best_step, elapsed_h())
        if val_loss < best_val - args.min_delta:
            best_val, best_step, bad_evals = val_loss, step, 0
            if rank == 0:
                best_path = os.path.join(args.exp, "best.pth")
                save_ckpt(model, best_path)
                with open(os.path.join(args.exp, "best.json"), "w", encoding="utf-8") as f:
                    json.dump({"step": step, "epoch": epoch + 1, "val_loss": val_loss,
                               "val_acc": val_acc, "elapsed_h": elapsed_h()}, f, indent=2)
                # epoch_end_done=False even when micro_done == n_micro: the epoch-end block has not run yet
                save_train_state(args.exp, best_path, optimizer, scheduler, epoch, micro_done, step,
                                 best_val, best_step, bad_evals, geometry, epoch_end_done=False)
        else:
            bad_evals += 1
            if args.patience > 0 and bad_evals >= args.patience:
                LOG.info("early stopping: %d evaluations without improvement", bad_evals)
                stop = True

    LOG.info("starting training%s", f" at epoch {start_epoch + 1}, step {step}" if args.resume else "")
    for epoch in range(start_epoch, args.epochs):
        # best.pth written by the validation on the last micro-batch of start_epoch: its epoch-end block
        # (epoch_XXX.pth + boundary sidecar) is still owed even when no optimizer step is left
        pending_epoch_end = (epoch == start_epoch and resumed_epoch_end_pending and skip_micro >= micro_per_epoch)
        if step >= total_steps and not pending_epoch_end:
            break
        train_sampler.set_epoch(epoch)
        ep_loss, ep_acc, ep_n = 0.0, 0.0, 0
        n_micro = len(train_loader)
        micro_done = 0
        trained = False
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            if epoch == start_epoch and i < skip_micro:
                continue  # deterministic sampler order: replay past the resume point without a forward
            micro_done = i + 1
            trained = True
            is_step = ((i + 1) % accum == 0) or (i + 1 == n_micro)
            # micro-batches in this optimizer step's group (the trailing group may be shorter)
            n_in_group = min(accum, n_micro - (i // accum) * accum)
            batch = to_device(batch, device)
            # DDP gradient all-reduce only on the last micro-batch of the group
            with (contextlib.nullcontext() if is_step else model.no_sync()):
                loss, acc = forward_prop(batch, model, device, args.bf16)
                (loss / n_in_group).backward()
            li, ai = loss.item(), acc.item()
            run_loss += li
            run_acc += ai
            run_n += 1
            ep_loss += li
            ep_acc += ai
            ep_n += 1
            if not is_step:
                continue
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                LOG.info("epoch %d step %d/%d (micro %d/%d): lr = %.8f, train_loss = %.4f, "
                         "train_acc = %.4f, time = %.3f h", epoch + 1, step, total_steps, micro_done,
                         n_micro, scheduler.get_last_lr()[0], run_loss / run_n, run_acc / run_n,
                         elapsed_h())
                run_loss, run_acc, run_n = 0.0, 0.0, 0
            if step % args.eval_every == 0:
                run_eval(epoch, micro_done)
            if stop or step >= total_steps:
                break

        # every micro-batch was consumed before the resume point and the epoch-end block never ran (best.pth
        # written by the validation on the last step of the epoch): run it now, without a second validation
        pending_end = not trained and pending_epoch_end
        if not trained and not pending_end:  # resumed past the end of an epoch already cut by --max_steps
            LOG.info("epoch %d: nothing left to train after the resume point", epoch + 1)
            continue
        if pending_end:
            LOG.info("epoch %d: all %d micro-batches were consumed before the resume point; running the pending "
                     "epoch-end checkpoint (validation already done at step %d)", epoch + 1, n_micro, step)
            micro_done = n_micro
        # end of epoch (or early exit): always validate and report like train.py
        elif step % args.eval_every != 0:
            run_eval(epoch, micro_done)
        LOG.info("=" * 25 + f"  Epoch {epoch + 1}/{args.epochs}, step {step}, LR: "
                 f"{scheduler.get_last_lr()[0]:.8f}, Time: {elapsed_h():.3f} h  " + "=" * 25)
        LOG.info("train_loss = %.4f, train_acc = %.4f", ep_loss / max(ep_n, 1), ep_acc / max(ep_n, 1))
        LOG.info("best val_loss = %.4f @ step %d", best_val, best_step)
        LOG.info("=" * 50)
        epoch_complete = micro_done >= n_micro
        last_epoch = stop or step >= total_steps or epoch + 1 == args.epochs
        if rank == 0 and (args.save_every_epoch or last_epoch):
            ep_path = os.path.join(args.exp, f"epoch_{epoch + 1:03d}.pth")
            save_ckpt(model, ep_path)
            # a --max_steps / early-stop cut leaves the epoch unfinished: resume re-enters it
            # after micro_done batches; a completed epoch resumes at the next one
            if epoch_complete:
                save_train_state(args.exp, ep_path, optimizer, scheduler, epoch + 1, 0, step,
                                 best_val, best_step, bad_evals, geometry, epoch_end_done=True)
            else:
                save_train_state(args.exp, ep_path, optimizer, scheduler, epoch, micro_done, step,
                                 best_val, best_step, bad_evals, geometry, epoch_end_done=True)
        if last_epoch:
            break

    LOG.info("done: %d steps, best val_loss %.4f @ step %d, %.3f h", step, best_val, best_step,
             elapsed_h())
    dist.barrier()
    dist.destroy_process_group()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if "LOCAL_RANK" not in os.environ:
        raise SystemExit("launch with torchrun (LOCAL_RANK not set); see module docstring")
    train(args)


if __name__ == "__main__":
    main()
