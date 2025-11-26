"""Data loading utilities for wejepa."""
from datasets import load_dataset
from huggingface_hub import snapshot_download
from .cifar import (
    IJEPACIFARDataset,
    build_train_transform as build_cifar_train_transform,
    build_eval_transform as build_cifar_eval_transform,
    create_pretraining_dataloader as create_cifar_pretraining_dataloader,
)
from .cub200 import (
    CUB200Dataset,
    build_cub_train_transform,
    build_cub_eval_transform,
    create_pretraining_dataloader as create_cub_pretraining_dataloader,
    create_classification_dataloaders,
    create_cub_inference_dataloader,
)
from .hf import (
    IJEPAHFDataset,
    build_train_transform as build_hf_train_transform,
    create_pretraining_dataloader as create_hf_pretraining_dataloader,
)
from .loader import create_pretraining_dataloader
from .feature_extractor import FeatureExtractor
from .image_dataset import ImageDataset

__all__ = [
    "CUB200Dataset",
    "IJEPACIFARDataset",
    "IJEPAHFDataset",
    "build_cifar_eval_transform",
    "build_cifar_train_transform",
    "build_cub_eval_transform",
    "build_cub_train_transform",
    "build_hf_train_transform",
    "create_cifar_pretraining_dataloader",
    "create_cub_classification_dataloaders",
    "create_cub_inference_dataloader",
    "create_cub_pretraining_dataloader",
    "create_hf_pretraining_dataloader",
    "create_pretraining_dataloader",
    "FeatureExtractor",
    "ImageDataset",
]
