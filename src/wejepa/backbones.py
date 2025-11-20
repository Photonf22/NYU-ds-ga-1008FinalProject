"""Reusable torchvision backbones."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch.nn as nn
from torchvision import transforms
from torchvision.models import (
    ViT_B_16_Weights,
    ViT_L_16_Weights,
    ConvNeXt_Tiny_Weights,
    ResNeXt50_32X4D_Weights,
    ResNet50_Weights,
    Swin_T_Weights,
    convnext_tiny,
    resnet50,
    resnext50_32x4d,
    swin_t,
    vit_b_16,
    vit_l_16,
)


@dataclass(frozen=True)
class BackboneSpec:
    """Metadata describing a torchvision backbone."""

    build_fn: Callable[..., nn.Module]
    weights: Optional[object]
    head_type: str
    default_image_size: int = 224


_BACKBONES: Dict[str, BackboneSpec] = {
    "vit_b_16": BackboneSpec(
        build_fn=vit_b_16,
        weights=ViT_B_16_Weights.IMAGENET1K_V1,
        head_type="vit",
    ),
    "vit_l_16": BackboneSpec(
        build_fn=vit_l_16,
        weights=ViT_L_16_Weights.IMAGENET1K_V1,
        head_type="vit",
    ),
    "resnet50": BackboneSpec(
        build_fn=resnet50,
        weights=ResNet50_Weights.IMAGENET1K_V2,
        head_type="fc",
    ),
    "resnext50_32x4d": BackboneSpec(
        build_fn=resnext50_32x4d,
        weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1,
        head_type="fc",
    ),
    "convnext_tiny": BackboneSpec(
        build_fn=convnext_tiny,
        weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        head_type="convnext",
    ),
    "swin_t": BackboneSpec(
        build_fn=swin_t,
        weights=Swin_T_Weights.IMAGENET1K_V1,
        head_type="swin",
    ),
}


def available_backbones() -> List[str]:
    """Return the list of registered backbone identifiers."""

    return sorted(_BACKBONES.keys())


def _get_classifier(model: nn.Module, head_type: str) -> nn.Module:
    if head_type == "vit":
        return model.heads.head  # type: ignore[return-value]
    if head_type == "fc":
        return model.fc  # type: ignore[return-value]
    if head_type == "convnext":
        return model.classifier[2]  # type: ignore[index]
    if head_type == "swin":
        return model.head  # type: ignore[return-value]
    raise ValueError(f"Unsupported head type: {head_type}")


def _set_classifier(model: nn.Module, head_type: str, new_head: nn.Module) -> None:
    if head_type == "vit":
        model.heads.head = new_head  # type: ignore[assignment]
    elif head_type == "fc":
        model.fc = new_head  # type: ignore[assignment]
    elif head_type == "convnext":
        classifier = model.classifier  # type: ignore[attr-defined]
        classifier[2] = new_head
    elif head_type == "swin":
        model.head = new_head  # type: ignore[assignment]
    else:
        raise ValueError(f"Unsupported head type: {head_type}")


def build_backbone(
    name: str,
    *,
    num_classes: Optional[int] = None,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    **kwargs,
) -> Tuple[nn.Module, int]:
    """Instantiate a torchvision backbone and optionally replace the classifier head.

    Parameters
    ----------
    name:
        Registry key identifying the backbone (``available_backbones`` lists choices).
    num_classes:
        Size of the classifier head.  If ``None`` the original number of classes is
        preserved.
    pretrained:
        Whether to load default ImageNet weights.  If ``False`` weights are randomly
        initialized.
    freeze_backbone:
        When ``True`` all parameters except the classifier head are frozen, making
        linear probing or pseudo-label inference easier.
    kwargs:
        Extra keyword arguments forwarded to the torchvision builder.
    """

    if name not in _BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Available: {available_backbones()}")
    spec = _BACKBONES[name]
    weights = spec.weights if pretrained else None
    model = spec.build_fn(weights=weights, **kwargs)
    classifier = _get_classifier(model, spec.head_type)
    feature_dim = classifier.in_features  # type: ignore[attr-defined]
    if num_classes is not None and classifier.out_features != num_classes:  # type: ignore[attr-defined]
        new_head = nn.Linear(feature_dim, num_classes)
        _set_classifier(model, spec.head_type, new_head)
    if freeze_backbone:
        classifier = _get_classifier(model, spec.head_type)
        head_params: Iterable[nn.Parameter] = classifier.parameters()
        head_param_ids = {id(p) for p in head_params}
        for param in model.parameters():
            if id(param) not in head_param_ids:
                param.requires_grad = False
    return model, feature_dim


def resolve_preprocess_transforms(
    name: str,
    *,
    pretrained: bool = True,
    image_size: Optional[int] = None,
) -> transforms.Compose:
    """Return image transforms matching the backbone's expected resolution."""

    if name not in _BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Available: {available_backbones()}")
    spec = _BACKBONES[name]
    if pretrained and spec.weights is not None:
        weight_enum = spec.weights
        return weight_enum.transforms()
    size = image_size or spec.default_image_size
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


__all__ = ["available_backbones", "build_backbone", "resolve_preprocess_transforms"]
