"""
CUB-200 dataset helpers maintained for backwards compatibility.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torchvision.transforms as T
from torch.utils.data import DataLoader

from ..config import IJepaConfig
from .standard import (
    StandardImageDataset,
    build_standard_eval_transform,
    build_standard_train_transform,
    create_classification_dataloaders as _standard_classification_dataloaders,
    create_inference_dataloader as _standard_inference_dataloader,
    create_pretraining_dataloader as _standard_pretrain_loader,
    ensure_standard_metadata,
)


class CUB200Dataset(StandardImageDataset):

    @staticmethod
    def _try_generate_cub200_metadata(dataset_root: Path):
        ensure_standard_metadata(dataset_root, "cub200")

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Optional[T.Compose] = None,
        *,
        return_labels: bool = False,
        return_filenames: bool = False,
        fake_size: Optional[int] = None,
        image_size: int = 96,
    ) -> None:
        super().__init__(
            root,
            split,
            dataset_name="cub200",
            transform=transform,
            return_labels=return_labels,
            return_filenames=return_filenames,
            fake_size=fake_size,
            image_size=image_size,
        )


def build_cub_train_transform(cfg: IJepaConfig) -> T.Compose:
    return build_standard_train_transform(cfg)


def build_cub_eval_transform(cfg: IJepaConfig) -> T.Compose:
    return build_standard_eval_transform(cfg)


def _cub_loader_kwargs(cfg: IJepaConfig, *, batch_size: int, shuffle: bool, sampler=None):
    from .standard import _loader_kwargs

    return _loader_kwargs(cfg, batch_size=batch_size, shuffle=shuffle, sampler=sampler)


def create_pretraining_dataloader(
    cfg: IJepaConfig,
    rank: int = 0,
    world_size: int = 1,
    split: str = "train",
):
    """Return a DataLoader for CUB-200 with the shared CSV-backed layout."""
    return _standard_pretrain_loader(cfg, rank=rank, world_size=world_size, split=split)


def create_classification_dataloaders(
    cfg: IJepaConfig, *, batch_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    return _standard_classification_dataloaders(cfg, batch_size=batch_size)


def create_cub_inference_dataloader(
    cfg: IJepaConfig,
    *,
    split: str = "test",
    batch_size: Optional[int] = None,
) -> DataLoader:
    return _standard_inference_dataloader(cfg, split=split, batch_size=batch_size)

create_cub_pretraining_dataloader = create_pretraining_dataloader
create_cub_classification_dataloaders = create_classification_dataloaders

__all__ = [
    "CUB200Dataset",
    "build_cub_train_transform",
    "build_cub_eval_transform",
    "create_pretraining_dataloader",
    "create_cub_pretraining_dataloader",
    "create_classification_dataloaders",
    "create_cub_classification_dataloaders",
    "create_cub_inference_dataloader",
]
