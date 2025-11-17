"""CIFAR-100 data helpers used for pretraining."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as T
import torch.distributed as dist

from ..config import IJepaConfig


class IJEPADataset(Dataset):
    """Return unlabeled images suitable for JEPA-style training."""

    def __init__(
        self,
        cfg: IJepaConfig,
        train: bool = True,
        transform: Optional[T.Compose] = None,
        download: bool = False,
    ) -> None:
        self.cfg = cfg
        self.transform = transform or (
            build_train_transform(cfg) if train else build_eval_transform(cfg)
        )
        image_size = cfg.data.image_size
        if cfg.data.use_fake_data:
            self.dataset: Dataset = torchvision.datasets.FakeData(
                size=cfg.data.fake_data_size,
                image_size=(3, image_size, image_size),
                num_classes=cfg.data.fake_data_size,
                transform=self.transform,
            )
        else:
            self.dataset = torchvision.datasets.CIFAR100(
                root=cfg.data.dataset_root,
                train=train,
                transform=self.transform,
                download=download,
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        img, *_ = self.dataset[index]
        # both the unlabeled CIFAR dataset and FakeData return (img, label)
        # but pretraining only consumes the image tensor.
        if isinstance(img, tuple):
            img = img[0]
        return img


def build_train_transform(cfg: IJepaConfig) -> T.Compose:
    dcfg = cfg.data
    transforms = [T.RandomResizedCrop(dcfg.image_size, scale=dcfg.crop_scale)]
    if dcfg.use_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if dcfg.use_color_distortion:
        transforms.append(
            T.ColorJitter(
                0.8 * dcfg.color_jitter,
                0.8 * dcfg.color_jitter,
                0.8 * dcfg.color_jitter,
                0.2 * dcfg.color_jitter,
            )
        )
    transforms.extend(
        [
            T.ToTensor(),
            T.Normalize(dcfg.normalization_mean, dcfg.normalization_std),
        ]
    )
    return T.Compose(transforms)


def build_eval_transform(cfg: IJepaConfig) -> T.Compose:
    dcfg = cfg.data
    return T.Compose(
        [
            T.Resize(dcfg.image_size),
            T.ToTensor(),
            T.Normalize(dcfg.normalization_mean, dcfg.normalization_std),
        ]
    )


def _worker_init_fn(worker_id: int) -> None:
    # dataloader worker seed setup to avoid duplication
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def create_pretraining_dataloader(
    cfg: IJepaConfig,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    dataset = IJEPADataset(cfg, train=True, download=rank == 0)
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()
    sampler: Optional[DistributedSampler] = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
    batch_size = cfg.data.train_batch_size
    if world_size > 1:
        batch_size = max(1, batch_size // world_size)
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
    )
    if cfg.data.num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.data.prefetch_factor

    loader = DataLoader(**loader_kwargs)
    return loader, sampler


__all__ = [
    "IJEPADataset",
    "build_train_transform",
    "build_eval_transform",
    "create_pretraining_dataloader",
]
