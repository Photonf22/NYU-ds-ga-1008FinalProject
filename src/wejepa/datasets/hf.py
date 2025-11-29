"""HuggingFace dataset helpers for pretraining."""
import importlib
import random
import numpy as np
from PIL import Image
from scipy import io
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T
import torch.distributed as dist

from ..config import IJepaConfig

class IJEPAHFDataset(Dataset):
    """Return unlabeled images from HuggingFace datasets."""

    def __init__(
        self,
        cfg: IJepaConfig,
        split: str = "train",
        transform: T.Compose = None,
    ):
        self.cfg = cfg
        self.transform = transform or build_train_transform(cfg)
        datasets = importlib.import_module("datasets")
        if cfg.data.dataset_name.lower() == "imagefolder":
            dataset_dir = getattr(cfg.data, "dataset_dir", None)
            if dataset_dir is None:
                raise ValueError("For 'imagefolder' dataset_name, 'dataset_dir' must be specified in the config.")

            dataset_dir = cfg.data.dataset_root + "/" + dataset_dir
            print(f"Loading imagefolder dataset from directory: {dataset_dir}")
            self.dataset = datasets.load_dataset(
                "imagefolder",
                data_dir=dataset_dir,
                split=split,
                cache_dir=cfg.data.dataset_root,
            )
        else:
            load_dataset = datasets.load_dataset
            self.dataset = load_dataset(
                cfg.data.dataset_name,
                split=split,
                cache_dir=cfg.data.dataset_root,
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        img = sample.get("image", None)
        if img is None:
            raise ValueError(f"No 'image' field in sample: {sample.keys()}")
        if isinstance(img, dict):
            print("Image dict keys:", img.keys())
            if "bytes" in img:
                img = Image.open(io.BytesIO(img["bytes"]))
            else:
                raise TypeError(f"Unknown image dict format: {img}")
        img = img.convert("RGB")
        img = self.transform(img)
        return img

def build_train_transform(cfg: IJepaConfig) -> T.Compose:
    dcfg = cfg.data
    transforms = [T.Resize((dcfg.image_size, dcfg.image_size))]
    transforms.extend([
        T.ToTensor(),
        T.Normalize(dcfg.normalization_mean, dcfg.normalization_std),
    ])
    return T.Compose(transforms)

def _worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

def create_pretraining_dataloader(
    cfg: IJepaConfig,
    rank: int = 0,
    world_size: int = 1,
    split: str = "train",
):
    dataset = IJEPAHFDataset(cfg, split=split)
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

__all__ = [
    "IJEPAHFDataset",
    "build_train_transform",
    "create_pretraining_dataloader",
]
