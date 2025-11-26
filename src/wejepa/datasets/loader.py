"""Generic dataset loader for pretraining."""

from __future__ import annotations
from typing import Tuple
from ..config import IJepaConfig

def create_pretraining_dataloader(cfg: IJepaConfig, rank: int = 0, world_size: int = 1):
    dataset_name = getattr(cfg.data, "dataset_name", None)
    if dataset_name is None or dataset_name.lower() == "cifar100":
        from wejepa.datasets.cifar import create_pretraining_dataloader as cifar_loader
        return cifar_loader(cfg, rank=rank, world_size=world_size)
    if dataset_name.lower() == "cub200":
        from wejepa.datasets.cub200 import create_pretraining_dataloader as cub200_loader
        return cub200_loader(cfg, rank=rank, world_size=world_size)
    else:
        from wejepa.datasets.hf import create_pretraining_dataloader as hf_loader
        return hf_loader(cfg, rank=rank, world_size=world_size, split="train")

__all__: Tuple[str, ...] = ("create_pretraining_dataloader",)
