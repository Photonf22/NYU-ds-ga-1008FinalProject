from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _default_dataset_root() -> str:
    return str(Path.cwd() / "data")


def _default_output_dir() -> str:
    return str(Path.cwd() / "outputs" / "ijepa")


@dataclass
class DataConfig:
    """Data-related hyperparameters."""

    dataset_root: str = field(default_factory=_default_dataset_root)
    dataset_name: str = "cifar100"
    dataset_dir: Optional[str] = None
    image_size: int = 32
    train_batch_size: int = 256
    eval_batch_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    crop_scale: Tuple[float, float] = (0.6, 1.0)
    color_jitter: Optional[float] = None
    use_color_distortion: Optional[bool] = False
    use_horizontal_flip: Optional[bool] = False
    normalization_mean: Tuple[float, float, float] = (0.5071, 0.4867, 0.4408)
    normalization_std: Tuple[float, float, float] = (0.2675, 0.2565, 0.2761)
    #normalization_mean: Optional[Tuple[float, float, float]] = None
    #normalization_std: Optional[Tuple[float, float, float]] = None
    #color_jitter: float = 0.5
    #use_color_distortion: bool = True
    #use_horizontal_flip: bool = True
    use_fake_data: bool = False
    fake_data_size: int = 512
    # For custom list-based datasets
    image_dir: Optional[str] = None
    image_list: Optional[str] = None
    labels: Optional[str] = None


@dataclass
class MaskConfig:
    """Target/context sampling hyperparameters."""

    target_aspect_ratio: Tuple[float, float] = (0.75, 1.5)
    target_scale: Tuple[float, float] = (0.15, 0.2)
    context_aspect_ratio: float = 1.0
    context_scale: Tuple[float, float] = (0.85, 1.0)
    num_target_blocks: int = 4


@dataclass
class ModelConfig:
    """Architecture settings"""

    img_size: int = 32
    patch_size: int = 4
    in_chans: int = 3
    embed_dim: int = 192
    enc_depth: int = 6
    pred_depth: int = 4
    num_heads: int = 6
    post_emb_norm: bool = False
    layer_dropout: float = 0.0
    classification_backbone: Optional[str] = None
    classification_num_classes: int = 100
    classification_pretrained: bool = False
    model_bypass: bool = False


@dataclass
class OptimizerConfig:
    """Optimization hyperparameters."""

    epochs: int = 5
    warmup_epochs: int = 1
    base_learning_rate: float = 1e-3
    start_learning_rate: float = 1e-4
    final_learning_rate: float = 1e-5
    weight_decay: float = 0.05
    final_weight_decay: float = 0.2
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip_norm: Optional[float] = 1.0
    momentum_teacher: float = 0.996
    momentum_teacher_final: float = 1.0


@dataclass
class HardwareConfig:
    """Execution environment settings."""

    seed: int = 42
    world_size: Optional[int] = None
    mixed_precision: bool = True
    compile_model: bool = False
    log_every: int = 50
    checkpoint_every: int = 1
    output_dir: str = field(default_factory=_default_output_dir)


@dataclass
class IJepaConfig:
    """Bundle all configuration sections."""

    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": asdict(self.data),
            "mask": asdict(self.mask),
            "model": asdict(self.model),
            "optimizer": asdict(self.optimizer),
            "hardware": asdict(self.hardware),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "IJepaConfig":
        data = DataConfig(**payload["data"])
        mask = MaskConfig(**payload["mask"]) if "mask" in payload else MaskConfig()
        model = ModelConfig(**payload["model"]) if "model" in payload else ModelConfig()
        optimizer = OptimizerConfig(**payload["optimizer"]) if "optimizer" in payload else OptimizerConfig()
        hardware = HardwareConfig(**payload["hardware"]) if "hardware" in payload else HardwareConfig()
        return cls(
            data=data,
            mask=mask,
            model=model,
            optimizer=optimizer,
            hardware=hardware,
        )

    def summary(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2)


def default_config() -> IJepaConfig:
    cfg = IJepaConfig()
    if not Path(cfg.data.dataset_root).exists():
        Path(cfg.data.dataset_root).mkdir(parents=True, exist_ok=True)
    return cfg


def hf224_config() -> IJepaConfig:
    """Configuration tailored for 224x224 Hugging Face image datasets."""

    cfg = default_config()
    cfg.data.dataset_name = "imagefolder"
    cfg.data.dataset_dir = "tsbpp_fall2025_deeplearning"
    cfg.data.image_size = 224
    cfg.model.img_size = 224
    cfg.model.patch_size = 16
    return cfg


__all__ = [
    "IJepaConfig",
    "DataConfig",
    "MaskConfig",
    "ModelConfig",
    "OptimizerConfig",
    "HardwareConfig",
    "default_config",
    "hf224_config",
]
