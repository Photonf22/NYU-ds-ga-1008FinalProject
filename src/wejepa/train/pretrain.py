"""
Pretraining entrypoint
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP

from ..config import IJepaConfig, default_config
from ..datasets import create_pretraining_dataloader
from ..model import IJEPA_base

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
Track avg and current value of a metric over time.
"""
class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


@dataclass
class TrainState:
    step: int = 0


def _set_seed(seed: int, rank: int) -> None:
    combined = seed + rank
    random.seed(combined)
    np.random.seed(combined)
    torch.manual_seed(combined)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(combined)


def _cosine_schedule(
    base: float,
    final: float,
    total_steps: int,
    warmup_steps: int = 0,
    start_value: Optional[float] = None,
) -> np.ndarray:
    if total_steps == 0:
        return np.zeros(0)
    schedule = np.zeros(total_steps, dtype=np.float32)
    start_value = base if start_value is None else start_value
    for step in range(total_steps):
        if step < warmup_steps and warmup_steps > 0:
            pct = step / warmup_steps
            schedule[step] = start_value + pct * (base - start_value)
        else:
            pct = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            schedule[step] = final + 0.5 * (base - final) * (1 + math.cos(math.pi * pct))
    return schedule


def _build_model(cfg: IJepaConfig, debug: bool = False) -> IJEPA_base:
    mcfg = cfg.model
    model = IJEPA_base(
        img_size=mcfg.img_size,
        patch_size=mcfg.patch_size,
        in_chans=mcfg.in_chans,
        embed_dim=mcfg.embed_dim,
        enc_depth=mcfg.enc_depth,
        pred_depth=mcfg.pred_depth,
        num_heads=mcfg.num_heads,
        post_emb_norm=mcfg.post_emb_norm,
        M=cfg.mask.num_target_blocks,
        layer_dropout=mcfg.layer_dropout,
        backbone=mcfg.classification_backbone,
        pretrained=mcfg.classification_pretrained,
        use_jepa_pos_with_backbone=mcfg.use_jepa_pos_with_backbone,
        debug=debug,
    )
    if debug:
        print(
            "[DEBUG] Built model with",
            f"img_size={mcfg.img_size}, patch_size={mcfg.patch_size}, embed_dim={mcfg.embed_dim},",
            f"enc_depth={mcfg.enc_depth}, pred_depth={mcfg.pred_depth}, num_heads={mcfg.num_heads}",
        )
    return model


def _setup_distributed(rank: int, world_size: int) -> None:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    device_id = rank if backend == "nccl" and torch.cuda.is_available() else None
    if device_id is not None:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size, device_id=device_id)
    else:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(
    module: IJEPA_base,
    optimizer: torch.optim.Optimizer,
    state: TrainState,
    epoch: int,
    cfg: IJepaConfig,
    debug: bool = False,
) -> Path:
    output_dir = Path(cfg.hardware.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch + 1,
        "step": state.step,
        "student": module.student_encoder.state_dict(),
        "teacher": module.teacher_encoder.state_dict(),
        "predictor": module.predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    ckpt_path = output_dir / f"ijepa_epoch_{epoch + 1:04d}.pt"
    torch.save(ckpt, ckpt_path)
    if debug:
        print(
            "[DEBUG] Saved checkpoint components:",
            {
                "epoch": epoch + 1,
                "step": state.step,
                "student": len(ckpt["student"]),
                "teacher": len(ckpt["teacher"]),
                "predictor": len(ckpt["predictor"]),
            },
        )
    return ckpt_path


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    lr_schedule: np.ndarray,
    wd_schedule: np.ndarray,
    momentum_schedule: np.ndarray,
    epoch: int,
    state: TrainState,
    cfg: IJepaConfig,
    device: torch.device,
    rank: int,
    debug: bool = False,
) -> Dict[str, float]:
    model.train()
    criterion = torch.nn.MSELoss()
    loss_meter = AverageMeter()
    start_time = time.time()
    module = model.module if isinstance(model, DDP) else model
    accum = 1
    for itr, images in enumerate(tqdm(data_loader, desc="Training", total=len(data_loader), dynamic_ncols=True)):
        images = images.to(device, non_blocking=True)
        schedule_idx = state.step
        if debug and itr == 0 and rank == 0:
            print(
                f"[DEBUG] Starting epoch {epoch + 1} with batch shape {tuple(images.shape)} "
                f"and schedule idx {schedule_idx}"
            )
        if schedule_idx < len(lr_schedule):
            lr = float(lr_schedule[schedule_idx])
            wd = float(wd_schedule[schedule_idx])
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
                param_group["weight_decay"] = wd
            if debug and itr % max(1, cfg.hardware.log_every) == 0 and rank == 0:
                print(
                    f"[DEBUG] Step {schedule_idx}: lr={lr:.6f}, weight_decay={wd:.6f}"
                )
        target_aspect_ratio = random.uniform(*cfg.mask.target_aspect_ratio)
        target_scale = random.uniform(*cfg.mask.target_scale)
        context_scale = random.uniform(*cfg.mask.context_scale)
        context_aspect_ratio = cfg.mask.context_aspect_ratio
        use_amp = cfg.hardware.mixed_precision and device.type == "cuda"
        dtype = torch.bfloat16 if use_amp else None  # or torch.float16 if you prefer
        if debug and itr == 0 and rank == 0:
            print(
                f"[DEBUG] Mask params: target_ar={target_aspect_ratio:.3f}, target_scale={target_scale:.3f}, "
                f"context_ar={context_aspect_ratio:.3f}, context_scale={context_scale:.3f}"
            )
            print(f"[DEBUG] Using AMP={use_amp} dtype={dtype}")
        with amp.autocast("cuda", enabled=use_amp, dtype=dtype):
            preds, targets = module(
                images,
                target_aspect_ratio=target_aspect_ratio,
                target_scale=target_scale,
                context_aspect_ratio=context_aspect_ratio,
                context_scale=context_scale,
            )
            loss = criterion(preds, targets) / accum
        if debug and itr % max(1, cfg.hardware.log_every) == 0 and rank == 0:
            print(
                f"[DEBUG] Iter {itr + 1}: preds={tuple(preds.shape)}, targets={tuple(targets.shape)}, "
                f"loss={loss.item() * accum:.4f}"
            )
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if scaler is not None:
            if cfg.optimizer.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.optimizer.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            if cfg.optimizer.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.optimizer.grad_clip_norm)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if schedule_idx < len(momentum_schedule):
            module.momentum_update(float(momentum_schedule[schedule_idx]))
        state.step += 1
        reduced_loss = loss.detach() * accum
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.AVG)
        loss_meter.update(reduced_loss.item(), n=images.size(0))
        if rank == 0 and (itr + 1) % cfg.hardware.log_every == 0:
            elapsed = time.time() - start_time
            total = cfg.data.train_batch_size * (itr + 1)
            ips = total / max(1e-6, elapsed)
            print(
                f"Epoch {epoch + 1} Iter {itr + 1}/{len(data_loader)} | Loss {loss_meter.avg:.4f} | {ips:.1f} img/s"
            )
    return {"loss": loss_meter.avg}


def _train_worker(rank: int, world_size: int, cfg_dict: Dict[str, Dict], debug: bool = False) -> None:
    cfg = IJepaConfig.from_dict(cfg_dict)
    _set_seed(cfg.hardware.seed, rank)
    # prepare device
    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    if debug and rank == 0:
        print(
            f"[DEBUG] Rank {rank} using device {device} | world_size={world_size} "
            f"| seed={cfg.hardware.seed}"
        )
    if world_size > 1:
        _setup_distributed(rank, world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    model = _build_model(cfg, debug=debug).to(device)
    if rank == 0:
        print(f"Model has {model.count_parameters():,} trainable parameters.")
    if cfg.hardware.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[attr-defined]
    if world_size > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    if debug and rank == 0:
        print(
            f"[DEBUG] Optimizer AdamW lr={cfg.optimizer.base_learning_rate} "
            f"betas={cfg.optimizer.betas} eps={cfg.optimizer.eps} "
            f"weight_decay={cfg.optimizer.weight_decay}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.base_learning_rate,
        betas=cfg.optimizer.betas,
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
    )

    resume_checkpoint = getattr(cfg, 'resume_checkpoint', None)
    if resume_checkpoint is not None and os.path.isfile(resume_checkpoint):
        if rank == 0:
            print(f"[DEBUG] Loading weights from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        # Load student, teacher, predictor weights
        module = model.module if isinstance(model, DDP) else model
        if 'student' in checkpoint:
            module.student_encoder.load_state_dict(checkpoint['student'])
        if 'teacher' in checkpoint:
            module.teacher_encoder.load_state_dict(checkpoint['teacher'])
        if 'predictor' in checkpoint:
            module.predictor.load_state_dict(checkpoint['predictor'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if rank == 0:
            print("[DEBUG] Successfully loaded checkpoint weights.")
    if debug and rank == 0:
        print(
            f"[DEBUG] Preparing dataloader with batch_size={cfg.data.train_batch_size}, "
            f"num_workers={cfg.data.num_workers}"
        )
    use_amp = cfg.hardware.mixed_precision and device.type == "cuda"
    scaler = amp.GradScaler("cuda", enabled=use_amp)
    data_loader, sampler = create_pretraining_dataloader(cfg, rank=rank, world_size=world_size)
    steps_per_epoch = len(data_loader)
    total_steps = steps_per_epoch * cfg.optimizer.epochs
    warmup_steps = cfg.optimizer.warmup_epochs * steps_per_epoch
    lr_schedule = _cosine_schedule(
        cfg.optimizer.base_learning_rate,
        cfg.optimizer.final_learning_rate,
        total_steps,
        warmup_steps,
        start_value=cfg.optimizer.start_learning_rate,
    )
    if debug and rank == 0:
        print(
            f"[DEBUG] total_steps={total_steps}, warmup_steps={warmup_steps}, steps_per_epoch={steps_per_epoch}"
        )
    wd_schedule = _cosine_schedule(
        cfg.optimizer.weight_decay,
        cfg.optimizer.final_weight_decay,
        total_steps,
        warmup_steps,
        start_value=cfg.optimizer.weight_decay,
    )
    momentum_schedule = _cosine_schedule(
        cfg.optimizer.momentum_teacher,
        cfg.optimizer.momentum_teacher_final,
        total_steps,
        warmup_steps=0,
        start_value=cfg.optimizer.momentum_teacher,
    )
    state = TrainState()
    if debug and rank == 0:
        print("[DEBUG] Starting training loop...")
    for epoch in range(cfg.optimizer.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        stats = train_one_epoch(
            model,
            data_loader,
            optimizer,
            scaler,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            epoch,
            state,
            cfg,
            device,
            rank,
            debug=debug,
        )
        print("Epoch completed.")
        if rank == 0 and (epoch + 1) % cfg.hardware.checkpoint_every == 0:
            module = model.module if isinstance(model, DDP) else model
            ckpt_path = save_checkpoint(module, optimizer, state, epoch, cfg, debug=debug)
            print(f"Saved checkpoint to {ckpt_path}")
        if rank == 0:
            print(f"Epoch {epoch + 1}/{cfg.optimizer.epochs} | loss={stats['loss']:.4f}")
    if world_size > 1:
        _cleanup_distributed()


def launch_pretraining(cfg: Optional[IJepaConfig] = None, debug: bool = False) -> None:
    cfg = cfg or default_config()
    world_size = cfg.hardware.world_size
    if world_size is None:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if world_size > 1:
        mp.spawn(_train_worker, args=(world_size, cfg.to_dict(), debug), nprocs=world_size, join=True)
    else:
        _train_worker(0, 1, cfg.to_dict(), debug)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the WE-JEPA encoder on requested dataset")
    parser.add_argument("--config", type=str, help="Path to a JSON config file", default=None)
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to a previous checkpoint to resume or initialize from (for domain adaptation or continued SSL)",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the default configuration and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debugging output for model setup and training",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.config:
        cfg_dict = json.loads(Path(args.config).read_text())
        cfg = IJepaConfig.from_dict(cfg_dict)
    else:
        cfg = default_config()
    # Add resume_checkpoint to config if provided
    if args.resume_checkpoint is not None:
        setattr(cfg, 'resume_checkpoint', args.resume_checkpoint)
    if args.print_config:
        print(cfg.summary())
        return

    if args.debug:
        print("[DEBUG] Loaded configuration:")
        print(cfg.summary())

    launch_pretraining(cfg, debug=args.debug)


if __name__ == "__main__":
    main()
