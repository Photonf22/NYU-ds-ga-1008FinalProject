"""Top-level package for wejepa."""
from .config import (
    IJepaConfig,
    DataConfig,
    MaskConfig,
    ModelConfig,
    OptimizerConfig,
    HardwareConfig,
    default_config,
)
from .backbones import available_backbones, build_backbone, resolve_preprocess_transforms
from .datasets import IJEPADataset, create_pretraining_dataloader
from .model import IJEPA_base
from .train.pretrain import launch_pretraining

__all__ = [
    "IJepaConfig",
    "DataConfig",
    "MaskConfig",
    "ModelConfig",
    "OptimizerConfig",
    "HardwareConfig",
    "default_config",
    "available_backbones",
    "build_backbone",
    "resolve_preprocess_transforms",
    "IJEPADataset",
    "create_pretraining_dataloader",
    "IJEPA_base",
    "launch_pretraining",
]

__version__ = "0.1.0"
