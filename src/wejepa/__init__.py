"""Top-level package for wejepa."""
from .config import (
    IJepaConfig,
    DataConfig,
    MaskConfig,
    ModelConfig,
    OptimizerConfig,
    HardwareConfig,
    default_config,
    hf224_config,
)
from .backbones import (
    adapt_config_for_backbone,
    available_backbones,
    build_backbone,
    choose_num_heads,
    get_backbone_spec,
    resolve_preprocess_transforms,
)
from .datasets import IJEPACIFARDataset, IJEPAHFDataset, create_pretraining_dataloader
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
    "hf224_config",
    "adapt_config_for_backbone",
    "available_backbones",
    "build_backbone",
    "choose_num_heads",
    "get_backbone_spec",
    "resolve_preprocess_transforms",
    "IJEPACIFARDataset",
    "IJEPAHFDataset",
    "create_pretraining_dataloader",
    "IJEPA_base",
    "launch_pretraining",
]

__version__ = "0.1.0"
