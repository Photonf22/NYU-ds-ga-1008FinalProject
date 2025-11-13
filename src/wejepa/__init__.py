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
from .data import IJEPADataset, create_pretraining_dataloader
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
    "IJEPADataset",
    "create_pretraining_dataloader",
    "IJEPA_base",
    "launch_pretraining",
]

__version__ = "0.1.0"
