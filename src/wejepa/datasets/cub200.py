"""CUB-200 dataset helpers for the Kaggle competition pipeline."""
from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as T

from ..config import IJepaConfig


@dataclass(frozen=True)
class CubSplit:
    name: str
    csv_name: str
    image_subdir: str


_SPLITS: Tuple[CubSplit, ...] = (
    CubSplit("train", "train_labels.csv", "train"),
    CubSplit("val", "val_labels.csv", "val"),
    CubSplit("test", "test_images.csv", "test"),
)


class CUB200Dataset(Dataset):
    """Dataset backed by the CSV files created by ``prepare_cub200_for_kaggle.py``.

    Parameters
    ----------
    root:
        Directory containing the ``train``, ``val``, and ``test`` folders along with
        the CSV metadata files.
    split:
        One of ``"train"``, ``"val"``, or ``"test"``.
    transform:
        Transform applied to each image.
    return_labels:
        When ``True`` the dataset returns a ``(image, label)`` tuple.
    return_filenames:
        When ``True`` the filename is appended to the returned tuple to support
        submission generation.
    fake_size:
        When ``cfg.data.use_fake_data`` is enabled this controls the number of
        synthetic samples to generate.
    """

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
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.return_labels = return_labels
        self.return_filenames = return_filenames
        self.fake_size = fake_size
        self.image_size = image_size

        if self.fake_size is not None:
            self._dataset = torchvision.datasets.FakeData(
                size=self.fake_size,
                image_size=(3, self.image_size, self.image_size),
                num_classes=max(self.fake_size, 200),
                transform=self.transform,
            )
            self.filenames: List[str] = [
                f"fake_{split}_{idx}.jpg" for idx in range(self.fake_size)
            ]
            return

        spec = _split_from_name(split)
        csv_path = self.root / spec.csv_name
        # If the CSV file does not exist, try to generate it from CUB-200-2011 raw files
        if not csv_path.exists():
            self._try_generate_cub200_metadata(self.root)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing metadata file {csv_path}. Could not auto-generate CUB-200 metadata. Please check your dataset root."
            )
        df = pd.read_csv(csv_path)
        self.filenames = df["filename"].tolist()
        self.labels: Optional[List[int]] = (
            df["class_id"].astype(int).tolist() if "class_id" in df.columns else None
        )
        self.image_dir = self.root / spec.image_subdir

    @staticmethod
    def _try_generate_cub200_metadata(dataset_root: Path):
        """
        Attempt to generate train/val/test CSVs and folders from CUB-200-2011 raw files if possible.
        """
        # Look for CUB_200_2011 subdir
        cub_dir = dataset_root / "CUB_200_2011"
        if not cub_dir.exists():
            # Try if dataset_root itself is the CUB_200_2011 dir
            cub_dir = dataset_root

        # Check for required files
        required_files = [
            cub_dir / 'images.txt',
            cub_dir / 'image_class_labels.txt',
            cub_dir / 'train_test_split.txt',
            cub_dir / 'classes.txt',
            cub_dir / 'images',
        ]
        if not all(f.exists() for f in required_files):
            return  # Cannot generate
        # Read metadata
        images_df = pd.read_csv(cub_dir / 'images.txt', sep=' ', names=['image_id', 'filepath'])
        labels_df = pd.read_csv(cub_dir / 'image_class_labels.txt', sep=' ', names=['image_id', 'class_id'])
        train_test_split_df = pd.read_csv(cub_dir / 'train_test_split.txt', sep=' ', names=['image_id', 'is_training_image'])
        classes_df = pd.read_csv(cub_dir / 'classes.txt', sep=' ', names=['class_id', 'class_name'])
        # Merge data
        data = images_df.merge(labels_df, on='image_id').merge(train_test_split_df, on='image_id')
        data = data.merge(classes_df, on='class_id')
        data['class_id'] = data['class_id'] - 1  # 0-indexed
        # Split data (use same ratios as script)
        np.random.seed(42)
        train_data, val_data, test_data = [], [], []
        for class_id in sorted(data['class_id'].unique()):
            class_data = data[data['class_id'] == class_id].copy()
            n_samples = len(class_data)
            class_data = class_data.sample(frac=1, random_state=42).reset_index(drop=True)
            n_train = int(n_samples * 0.7)
            n_val = int(n_samples * 0.15)
            train_data.append(class_data.iloc[:n_train])
            val_data.append(class_data.iloc[n_train:n_train+n_val])
            test_data.append(class_data.iloc[n_train+n_val:])
        train_df = pd.concat(train_data, ignore_index=True)
        val_df = pd.concat(val_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)
        # Copy images and build flat structure
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            split_dir = dataset_root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for idx, row in split_df.iterrows():
                src = cub_dir / 'images' / row['filepath']
                dst_filename = f"{row['image_id']:05d}_{Path(row['filepath']).name}"
                dst = split_dir / dst_filename
                if not dst.exists():
                    try:
                        shutil.copy2(src, dst)
                    except Exception:
                        continue
                split_df.at[idx, 'filename'] = dst_filename
        # Save metadata CSVs
        train_df[['filename', 'class_id', 'class_name']].to_csv(dataset_root / 'train_labels.csv', index=False)
        val_df[['filename', 'class_id', 'class_name']].to_csv(dataset_root / 'val_labels.csv', index=False)
        test_df[['filename']].to_csv(dataset_root / 'test_images.csv', index=False)

    def __len__(self) -> int:
        if hasattr(self, "_dataset"):
            return len(self._dataset)
        return len(self.filenames)

    def __getitem__(self, index: int):
        if hasattr(self, "_dataset"):
            image, label = self._dataset[index]
            items: List[object] = [image]
            if self.return_labels:
                items.append(int(label))
            if self.return_filenames:
                items.append(self.filenames[index])
            return tuple(items) if len(items) > 1 else items[0]

        filename = self.filenames[index]
        image_path = self.image_dir / filename
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        items = [image]
        if self.return_labels and self.labels is not None:
            items.append(int(self.labels[index]))
        if self.return_filenames:
            items.append(filename)
        return tuple(items) if len(items) > 1 else items[0]


def _split_from_name(name: str) -> CubSplit:
    for split in _SPLITS:
        if split.name == name:
            return split
    valid = ", ".join(s.name for s in _SPLITS)
    raise ValueError(f"Unknown split '{name}'. Expected one of: {valid}")


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_cub_train_transform(cfg: IJepaConfig) -> T.Compose:
    dcfg = cfg.data
    transforms: List[object] = [T.RandomResizedCrop(dcfg.image_size, scale=dcfg.crop_scale)]
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


def build_cub_eval_transform(cfg: IJepaConfig) -> T.Compose:
    dcfg = cfg.data
    return T.Compose(
        [
            T.Resize((dcfg.image_size, dcfg.image_size)),
            T.ToTensor(),
            T.Normalize(dcfg.normalization_mean, dcfg.normalization_std),
        ]
    )


def _cub_loader_kwargs(cfg: IJepaConfig, *, batch_size: int, shuffle: bool, sampler=None):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=shuffle,
        worker_init_fn=_worker_init_fn,
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
    )
    if cfg.data.num_workers > 0:
        kwargs["prefetch_factor"] = cfg.data.prefetch_factor
    return kwargs


def create_pretraining_dataloader(
    cfg: IJepaConfig,
    rank: int = 0,
    world_size: int = 1,
    split: str = "train",
):
    """Return a DataLoader for CUB-200, similar to hf.py."""
    transform = build_cub_train_transform(cfg)
    fake_size = cfg.data.fake_data_size if cfg.data.use_fake_data else None
    dataset = CUB200Dataset(
        cfg.data.dataset_root,
        split,
        transform=transform,
        return_labels=False,
        fake_size=fake_size,
        image_size=cfg.data.image_size,
    )
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()
    sampler = None
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


def create_classification_dataloaders(
    cfg: IJepaConfig, *, batch_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    train_transform = build_cub_train_transform(cfg)
    eval_transform = build_cub_eval_transform(cfg)
    fake_size = cfg.data.fake_data_size if cfg.data.use_fake_data else None

    train_ds = CUB200Dataset(
        cfg.data.dataset_root,
        "train",
        transform=train_transform,
        return_labels=True,
        fake_size=fake_size,
        image_size=cfg.data.image_size,
    )
    val_ds = CUB200Dataset(
        cfg.data.dataset_root,
        "val",
        transform=eval_transform,
        return_labels=True,
        fake_size=fake_size,
        image_size=cfg.data.image_size,
    )
    bsz = batch_size or cfg.data.train_batch_size
    train_loader = DataLoader(
        train_ds,
        **_cub_loader_kwargs(cfg, batch_size=bsz, shuffle=True),
    )
    val_loader = DataLoader(
        val_ds,
        **_cub_loader_kwargs(cfg, batch_size=cfg.data.eval_batch_size, shuffle=False),
    )
    return train_loader, val_loader


def create_cub_inference_dataloader(
    cfg: IJepaConfig,
    *,
    split: str = "test",
    batch_size: Optional[int] = None,
) -> DataLoader:
    transform = build_cub_eval_transform(cfg)
    fake_size = cfg.data.fake_data_size if cfg.data.use_fake_data else None
    dataset = CUB200Dataset(
        cfg.data.dataset_root,
        split,
        transform=transform,
        return_labels=False,
        return_filenames=True,
        fake_size=fake_size,
        image_size=cfg.data.image_size,
    )
    loader = DataLoader(
        dataset,
        **_cub_loader_kwargs(
            cfg, batch_size=batch_size or cfg.data.eval_batch_size, shuffle=False
        ),
    )
    return loader


# Backwards compatible aliases for exported helpers
create_cub_pretraining_dataloader = create_pretraining_dataloader
create_cub_classification_dataloaders = create_classification_dataloaders

__all__ = [
    "CUB200Dataset",
    "CubSplit",
    "build_cub_train_transform",
    "build_cub_eval_transform",
    "create_pretraining_dataloader",
    "create_cub_pretraining_dataloader",
    "create_classification_dataloaders",
    "create_cub_classification_dataloaders",
    "create_cub_inference_dataloader",
]
