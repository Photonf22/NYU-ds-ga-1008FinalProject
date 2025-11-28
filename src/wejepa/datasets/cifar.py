"""
CIFAR-100 data helpers.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torchvision.transforms as T

from ..config import IJepaConfig
from .standard import (
    StandardImageDataset,
    build_standard_eval_transform as build_eval_transform,
    build_standard_train_transform as build_train_transform,
    create_pretraining_dataloader as _standard_pretrain_loader,
)


class IJEPACIFARDataset(StandardImageDataset):

    def __init__(
        self,
        cfg: IJepaConfig,
        train: bool = True,
        transform: Optional[T.Compose] = None,
    ) -> None:
        super().__init__(
            cfg.data.dataset_root,
            "train" if train else "test",
            dataset_name=getattr(cfg.data, "dataset_name", "cifar100"),
            transform=transform or (build_train_transform(cfg) if train else build_eval_transform(cfg)),
            return_labels=False,
            fake_size=cfg.data.fake_data_size if cfg.data.use_fake_data else None,
            image_size=cfg.data.image_size,
        )


def create_pretraining_dataloader(
    cfg: IJepaConfig,
    rank: int = 0,
    world_size: int = 1,
):

    return _standard_pretrain_loader(cfg, rank=rank, world_size=world_size, split="train")


__all__: Tuple[str, ...] = (
    "IJEPACIFARDataset",
    "build_train_transform",
    "build_eval_transform",
    "create_pretraining_dataloader",
)
