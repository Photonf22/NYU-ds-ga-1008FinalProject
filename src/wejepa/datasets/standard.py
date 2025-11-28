"""
Dataset helpers
"""
from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as T

from ..config import IJepaConfig


@dataclass(frozen=True)
class StandardSplit:
    name: str
    csv_name: str
    image_subdir: str


STANDARD_SPLITS: Tuple[StandardSplit, ...] = (
    StandardSplit("train", "train_labels.csv", "train"),
    StandardSplit("val", "val_labels.csv", "val"),
    StandardSplit("test", "test_images.csv", "test"),
)


def _split_from_name(name: str) -> StandardSplit:
    for split in STANDARD_SPLITS:
        if split.name == name:
            return split
    valid = ", ".join(s.name for s in STANDARD_SPLITS)
    raise ValueError(f"Unknown split '{name}'. Expected one of: {valid}")


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_standard_train_transform(cfg: IJepaConfig) -> T.Compose:
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


def build_standard_eval_transform(cfg: IJepaConfig) -> T.Compose:
    dcfg = cfg.data
    return T.Compose(
        [
            T.Resize((dcfg.image_size, dcfg.image_size)),
            T.ToTensor(),
            T.Normalize(dcfg.normalization_mean, dcfg.normalization_std),
        ]
    )


def _write_standard_csvs(
    target_root: Path,
    train_df: Optional[pd.DataFrame],
    val_df: Optional[pd.DataFrame],
    test_df: Optional[pd.DataFrame],
):
    if train_df is not None and not train_df.empty:
        train_df[["filename", "class_id", "class_name"]].to_csv(
            target_root / "train_labels.csv", index=False
        )
    if val_df is not None and not val_df.empty:
        val_df[["filename", "class_id", "class_name"]].to_csv(
            target_root / "val_labels.csv", index=False
        )
    if test_df is not None and not test_df.empty:
        test_df[["filename"]].to_csv(target_root / "test_images.csv", index=False)
        sample_submission = pd.DataFrame({
            "filename": test_df["filename"],
            "class_id": 0,
        })
        sample_submission.to_csv(target_root / "sample_submission.csv", index=False)


def _generate_from_cub200_metadata(dataset_root: Path) -> bool:
    cub_dir = dataset_root / "CUB_200_2011"
    if not cub_dir.exists():
        cub_dir = dataset_root
    required_files = [
        cub_dir / "images.txt",
        cub_dir / "image_class_labels.txt",
        cub_dir / "train_test_split.txt",
        cub_dir / "classes.txt",
        cub_dir / "images",
    ]
    if not all(f.exists() for f in required_files):
        return False

    images_df = pd.read_csv(cub_dir / "images.txt", sep=" ", names=["image_id", "filepath"])
    labels_df = pd.read_csv(
        cub_dir / "image_class_labels.txt", sep=" ", names=["image_id", "class_id"]
    )
    train_test_split_df = pd.read_csv(
        cub_dir / "train_test_split.txt", sep=" ", names=["image_id", "is_training_image"]
    )
    classes_df = pd.read_csv(cub_dir / "classes.txt", sep=" ", names=["class_id", "class_name"])

    data = images_df.merge(labels_df, on="image_id").merge(train_test_split_df, on="image_id")
    data = data.merge(classes_df, on="class_id")
    data["class_id"] = data["class_id"] - 1

    np.random.seed(42)
    train_data: List[pd.DataFrame] = []
    val_data: List[pd.DataFrame] = []
    test_data: List[pd.DataFrame] = []
    for class_id in sorted(data["class_id"].unique()):
        class_data = data[data["class_id"] == class_id].copy()
        n_samples = len(class_data)
        class_data = class_data.sample(frac=1, random_state=42).reset_index(drop=True)
        n_train = int(n_samples * 0.7)
        n_val = int(n_samples * 0.15)
        train_data.append(class_data.iloc[:n_train])
        val_data.append(class_data.iloc[n_train : n_train + n_val])
        test_data.append(class_data.iloc[n_train + n_val :])

    train_df = pd.concat(train_data, ignore_index=True)
    val_df = pd.concat(val_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_dir = dataset_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for idx, row in split_df.iterrows():
            src = cub_dir / "images" / row["filepath"]
            dst_filename = f"{row['image_id']:05d}_{Path(row['filepath']).name}"
            dst = split_dir / dst_filename
            if not dst.exists():
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    continue
            split_df.at[idx, "filename"] = dst_filename

    _write_standard_csvs(dataset_root, train_df, val_df, test_df)
    return True


def _generate_from_cifar100(dataset_root: Path, val_ratio: float) -> bool:
    try:
        cifar_train = torchvision.datasets.CIFAR100(root=str(dataset_root), train=True, download=True)
        cifar_test = torchvision.datasets.CIFAR100(root=str(dataset_root), train=False, download=True)
    except Exception:
        return False

    def _export_split(dataset, split_name: str) -> pd.DataFrame:
        split_dir = dataset_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, object]] = []
        for idx, (img, label) in enumerate(dataset):
            filename = f"{split_name}_{idx:06d}.png"
            target_path = split_dir / filename
            if not target_path.exists():
                img.save(target_path)
            class_name = cifar_train.classes[label]
            rows.append({"filename": filename, "class_id": int(label), "class_name": class_name})
        return pd.DataFrame(rows)

    full_train = _export_split(cifar_train, "train_full")
    val_size = int(len(full_train) * val_ratio)
    val_df = full_train.sample(n=val_size, random_state=42)
    train_df = full_train.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    for df, split_dir in [(train_df, train_dir), (val_df, val_dir)]:
        for filename in df["filename"]:
            src = dataset_root / "train_full" / filename
            dst = split_dir / filename
            if not dst.exists() and src.exists():
                shutil.copy2(src, dst)
    if (dataset_root / "train_full").exists():
        shutil.rmtree(dataset_root / "train_full", ignore_errors=True)

    test_df = _export_split(cifar_test, "test")
    _write_standard_csvs(dataset_root, train_df, val_df, test_df)
    return True


def _generate_from_existing_folders(dataset_root: Path, val_ratio: float) -> bool:
    split_dirs = {split.name: dataset_root / split.image_subdir for split in STANDARD_SPLITS}
    if not any(path.exists() for path in split_dirs.values()):
        return False

    class_names = set()
    for split_dir in split_dirs.values():
        if not split_dir.exists():
            continue
        for subdir in split_dir.iterdir():
            if subdir.is_dir():
                class_names.add(subdir.name)
    class_name_list = sorted(class_names)
    class_to_idx = {name: idx for idx, name in enumerate(class_name_list)}

    def _index_split(split_name: str, split_dir: Path) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        if not split_dir.exists():
            return pd.DataFrame(rows)
        for path in split_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(split_dir)
            class_id = None
            class_name = None
            parts = rel.parts
            if len(parts) > 1:
                class_name = parts[0]
                class_id = class_to_idx.get(class_name)
            rows.append(
                {
                    "filename": str(rel),
                    "class_id": class_id,
                    "class_name": class_name,
                }
            )
        return pd.DataFrame(rows)

    train_dir = split_dirs.get("train", dataset_root / "train")
    val_dir = split_dirs.get("val", dataset_root / "val")
    test_dir = split_dirs.get("test", dataset_root / "test")

    train_df = _index_split("train", train_dir)
    val_df = _index_split("val", val_dir)
    test_df = _index_split("test", test_dir)

    if val_df.empty and not train_df.empty and val_ratio > 0:
        val_count = max(1, int(len(train_df) * val_ratio))
        val_df = train_df.sample(n=val_count, random_state=42).copy().reset_index(drop=True)
        remaining = train_df.drop(val_df.index).reset_index(drop=True)
        train_df = remaining
        val_dir.mkdir(parents=True, exist_ok=True)
        for fname in val_df["filename"]:
            src = train_dir / fname
            dst = val_dir / fname
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy2(src, dst)
            val_df.loc[val_df["filename"] == fname, "filename"] = str(Path(fname).as_posix())

    if train_df.empty and val_df.empty and test_df.empty:
        return False

    _write_standard_csvs(dataset_root, train_df if not train_df.empty else None, val_df if not val_df.empty else None, test_df if not test_df.empty else None)
    return True


def ensure_standard_metadata(
    dataset_root: Path,
    dataset_name: str,
    *,
    val_ratio: float = 0.1,
) -> None:
    target_root = Path(dataset_root)
    csvs_present = all((target_root / split.csv_name).exists() for split in STANDARD_SPLITS)
    if csvs_present:
        return

    dataset_key = dataset_name.lower().replace("_", "").replace("-", "")
    if dataset_key.startswith("cub"):
        if _generate_from_cub200_metadata(target_root):
            return
    if dataset_key.startswith("cifar100") or dataset_key.startswith("cifar"):
        if _generate_from_cifar100(target_root, val_ratio=val_ratio):
            return
    if _generate_from_existing_folders(target_root, val_ratio):
        return

    raise FileNotFoundError(
        f"Could not generate standard metadata at {target_root}. "
        f"Ensure the dataset is downloaded or supply CSVs manually."
    )


class StandardImageDataset(Dataset):

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        dataset_name: str,
        transform: Optional[T.Compose] = None,
        return_labels: bool = False,
        return_filenames: bool = False,
        fake_size: Optional[int] = None,
        image_size: int = 96,
        val_ratio: float = 0.1,
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
            self.filenames: List[str] = [f"fake_{split}_{idx}.jpg" for idx in range(self.fake_size)]
            return

        ensure_standard_metadata(self.root, dataset_name, val_ratio=val_ratio)
        spec = _split_from_name(split)
        csv_path = self.root / spec.csv_name
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing metadata file {csv_path}. Ensure_standard_metadata could not create it."
            )
        df = pd.read_csv(csv_path)
        self.filenames = df["filename"].tolist()
        self.labels: Optional[List[int]] = (
            df["class_id"].astype(int).tolist() if "class_id" in df.columns and not df["class_id"].isnull().all() else None
        )
        self.image_dir = self.root / spec.image_subdir

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


def _loader_kwargs(cfg: IJepaConfig, *, batch_size: int, shuffle: bool, sampler=None):
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
    *,
    rank: int = 0,
    world_size: int = 1,
    split: str = "train",
):
    transform = build_standard_train_transform(cfg)
    fake_size = cfg.data.fake_data_size if cfg.data.use_fake_data else None
    dataset_root = Path(cfg.data.dataset_root)
    if getattr(cfg.data, "dataset_dir", None):
        dataset_root = dataset_root / cfg.data.dataset_dir
    dataset = StandardImageDataset(
        dataset_root,
        split,
        dataset_name=getattr(cfg.data, "dataset_name", "cifar100"),
        transform=transform,
        return_labels=False,
        fake_size=fake_size,
        image_size=cfg.data.image_size,
        val_ratio=getattr(cfg.data, "val_split", 0.1),
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
    train_transform = build_standard_train_transform(cfg)
    eval_transform = build_standard_eval_transform(cfg)
    fake_size = cfg.data.fake_data_size if cfg.data.use_fake_data else None

    dataset_root = Path(cfg.data.dataset_root)
    if getattr(cfg.data, "dataset_dir", None):
        dataset_root = dataset_root / cfg.data.dataset_dir

    train_ds = StandardImageDataset(
        dataset_root,
        "train",
        dataset_name=getattr(cfg.data, "dataset_name", "cifar100"),
        transform=train_transform,
        return_labels=True,
        fake_size=fake_size,
        image_size=cfg.data.image_size,
        val_ratio=getattr(cfg.data, "val_split", 0.1),
    )
    val_ds = StandardImageDataset(
        dataset_root,
        "val",
        dataset_name=getattr(cfg.data, "dataset_name", "cifar100"),
        transform=eval_transform,
        return_labels=True,
        fake_size=fake_size,
        image_size=cfg.data.image_size,
        val_ratio=getattr(cfg.data, "val_split", 0.1),
    )
    bsz = batch_size or cfg.data.train_batch_size
    train_loader = DataLoader(
        train_ds,
        **_loader_kwargs(cfg, batch_size=bsz, shuffle=True),
    )
    val_loader = DataLoader(
        val_ds,
        **_loader_kwargs(cfg, batch_size=cfg.data.eval_batch_size, shuffle=False),
    )
    return train_loader, val_loader


def create_inference_dataloader(
    cfg: IJepaConfig,
    *,
    split: str = "test",
    batch_size: Optional[int] = None,
) -> DataLoader:
    transform = build_standard_eval_transform(cfg)
    fake_size = cfg.data.fake_data_size if cfg.data.use_fake_data else None
    dataset_root = Path(cfg.data.dataset_root)
    if getattr(cfg.data, "dataset_dir", None):
        dataset_root = dataset_root / cfg.data.dataset_dir
    dataset = StandardImageDataset(
        dataset_root,
        split,
        dataset_name=getattr(cfg.data, "dataset_name", "cifar100"),
        transform=transform,
        return_labels=False,
        return_filenames=True,
        fake_size=fake_size,
        image_size=cfg.data.image_size,
        val_ratio=getattr(cfg.data, "val_split", 0.1),
    )
    loader = DataLoader(
        dataset,
        **_loader_kwargs(
            cfg, batch_size=batch_size or cfg.data.eval_batch_size, shuffle=False
        ),
    )
    return loader


__all__ = [
    "StandardImageDataset",
    "StandardSplit",
    "build_standard_train_transform",
    "build_standard_eval_transform",
    "create_pretraining_dataloader",
    "create_classification_dataloaders",
    "create_inference_dataloader",
    "ensure_standard_metadata",
    "STANDARD_SPLITS",
    "_loader_kwargs",
]
