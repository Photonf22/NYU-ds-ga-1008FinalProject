"""
Generic dataset loader
"""

from __future__ import annotations
from typing import Tuple
from ..config import IJepaConfig

from .standard import create_pretraining_dataloader as _standard_pretrain


def create_pretraining_dataloader(cfg: IJepaConfig, rank: int = 0, world_size: int = 1):
    return _standard_pretrain(cfg, rank=rank, world_size=world_size, split="train")


__all__: Tuple[str, ...] = ("create_pretraining_dataloader",)
