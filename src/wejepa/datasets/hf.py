"""
HuggingFace dataset helpers for using the standard layout.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torchvision.transforms as T

from ..config import IJepaConfig
from .standard import (
    StandardImageDataset,
    build_standard_eval_transform as build_eval_transform,
    build_standard_train_transform as build_train_transform,
    create_pretraining_dataloader as _standard_pretrain_loader,
)


class IJEPAHFDataset(StandardImageDataset):

    def __init__(
        self,
        cfg: IJepaConfig,
        split: str = "train",
        transform: Optional[T.Compose] = None,
    ):
        dataset_root = cfg.data.dataset_root
        if getattr(cfg.data, "dataset_dir", None):
            dataset_root = str(Path(cfg.data.dataset_root) / cfg.data.dataset_dir)
        super().__init__(
            dataset_root,
            split,
            dataset_name=getattr(cfg.data, "dataset_name", "imagefolder"),
            transform=transform or (build_train_transform(cfg) if split == "train" else build_eval_transform(cfg)),
            return_labels=False,
            fake_size=cfg.data.fake_data_size if cfg.data.use_fake_data else None,
            image_size=cfg.data.image_size,
        )


def create_pretraining_dataloader(
    cfg: IJepaConfig,
    rank: int = 0,
    world_size: int = 1,
    split: str = "train",
):
    return _standard_pretrain_loader(cfg, rank=rank, world_size=world_size, split=split)


__all__: Tuple[str, ...] = (
    "IJEPAHFDataset",
    "build_train_transform",
    "build_eval_transform",
    "create_pretraining_dataloader",
)
