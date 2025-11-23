"""Training utilities exposed at the package level."""

from .finetune import (
    FinetuneConfig,
    FinetuneReport,
    LinearProbe,
    compare_pretrained_vs_scratch,
    create_finetune_dataloader,
    load_backbone_from_checkpoint,
    train_linear_probe,
)
# from .pretrain import launch_pretraining
from typing import TYPE_CHECKING

__all__ = [
    "launch_pretraining",
    "FinetuneConfig",
    "FinetuneReport",
    "LinearProbe",
    "compare_pretrained_vs_scratch",
    "create_finetune_dataloader",
    "load_backbone_from_checkpoint",
    "train_linear_probe",
]

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from .finetune import (
        FinetuneConfig,
        LinearProbe,
        create_finetune_dataloader,
        load_backbone_from_checkpoint,
        train_linear_probe,
    )
    from .pretrain import launch_pretraining


def __getattr__(name):
    if name == "launch_pretraining":
        from .pretrain import launch_pretraining

        return launch_pretraining

    finetune_attrs = {
        "FinetuneConfig",
        "LinearProbe",
        "create_finetune_dataloader",
        "load_backbone_from_checkpoint",
        "train_linear_probe",
    }

    if name in finetune_attrs:
        from . import finetune

        return getattr(finetune, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
