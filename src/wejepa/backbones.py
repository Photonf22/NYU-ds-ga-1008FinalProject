"""Wrapper for torchvision backbones."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import (
    ViT_B_16_Weights,
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
    Swin_T_Weights,
    Swin_S_Weights,
    convnext_tiny,
    convnext_small,
    swin_t,
    swin_s,
    vit_b_16,
)


@dataclass(frozen=True)
class BackboneSpec:
    """Metadata describing a torchvision backbone."""

    build_fn: Callable[..., nn.Module]
    weights: Optional[object]
    head_type: str
    default_image_size: int = 224
    default_patch_size: Optional[int] = None
    default_embed_dim: Optional[int] = None
    default_num_heads: Optional[int] = None


_BACKBONES: Dict[str, BackboneSpec] = {
    "vit_b_16": BackboneSpec(
        build_fn=vit_b_16,
        weights=ViT_B_16_Weights.IMAGENET1K_V1,
        head_type="vit",
        default_patch_size=16,
        default_embed_dim=768,
        default_num_heads=12,
    ),
    "convnext_tiny": BackboneSpec(
        build_fn=convnext_tiny,
        weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        head_type="convnext",
        default_patch_size=32,
        default_embed_dim=768,
        default_num_heads=12,
    ),
    "convnext_small": BackboneSpec(
        build_fn=convnext_small,
        weights=ConvNeXt_Small_Weights.IMAGENET1K_V1,
        head_type="convnext",
        default_patch_size=32,
        default_embed_dim=384,
        default_num_heads=6,
    ),
    "swin_t": BackboneSpec(
        build_fn=swin_t,
        weights=Swin_T_Weights.IMAGENET1K_V1,
        head_type="swin",
        default_patch_size=4,
        default_embed_dim=768,
        default_num_heads=12,
    ),
    "swin_s": BackboneSpec(
        build_fn=swin_s,
        weights=Swin_S_Weights.IMAGENET1K_V1,
        head_type="swin",
        default_patch_size=4,
        default_embed_dim=768,
        default_num_heads=12,
    ),
}


def available_backbones() -> List[str]:
    """Return the list of registered backbone identifiers."""

    return sorted(_BACKBONES.keys())


def get_backbone_spec(name: str) -> BackboneSpec:
    """Return the metadata for a registered backbone."""

    if name not in _BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Available: {available_backbones()}")
    return _BACKBONES[name]


def adapt_config_for_backbone(cfg: "IJepaConfig", backbone: str) -> "IJepaConfig":
    """Return a copy of ``cfg`` sized for the requested backbone.

    The helper keeps dataset and model image sizes in sync with the
    backbone's native resolution (e.g., 224 for ImageNet-pretrained
    architectures) and records the chosen backbone on the model config.
    """

    import copy

    spec = get_backbone_spec(backbone)
    updated = copy.deepcopy(cfg)
    updated.model.classification_backbone = backbone
    updated.model.img_size = spec.default_image_size
    updated.model.patch_size = spec.default_patch_size or updated.model.patch_size
    updated.data.image_size = spec.default_image_size

    # Align embedding dimension and number of heads with backbone defaults so the
    # transformer encoders can construct valid attention blocks.
    if spec.default_embed_dim is not None:
        updated.model.embed_dim = spec.default_embed_dim
    preferred_heads = spec.default_num_heads or updated.model.num_heads
    updated.model.num_heads = choose_num_heads(
        updated.model.embed_dim, preferred_heads, updated.model.num_heads
    )
    return updated


def choose_num_heads(embed_dim: int, *preferred: int) -> int:
    """Return a head count that divides ``embed_dim``.

    The function tries any provided preferred values first, then falls back to the
    largest divisor of `embed_dim` not exceeding 32.
    """

    for candidate in preferred:
        if candidate is not None and embed_dim % candidate == 0:
            return candidate
    for candidate in range(min(embed_dim, 32), 0, -1):
        if embed_dim % candidate == 0:
            return candidate
    return 1


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
    use_jepa_pos_with_backbone: bool = True,
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

    spec = get_backbone_spec(name)
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
    wrapped = _TokenizingBackbone(model, spec, use_jepa_pos_with_backbone=use_jepa_pos_with_backbone)
    return wrapped, feature_dim


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

class _TokenizingBackbone(nn.Module):
    """Wrap torchvision classification models to emit patch tokens."""

    def __init__(self, model: nn.Module, 
                 spec: BackboneSpec, 
                 use_jepa_pos_with_backbone: bool = True) -> None:
        super().__init__()
        self.model = model
        self.spec = spec
        self.use_jepa_pos_with_backbone = use_jepa_pos_with_backbone
        self.image_size = getattr(model, "image_size", spec.default_image_size)
        self.patch_size = getattr(model, "patch_size", spec.default_patch_size)
        self.hidden_dim = getattr(model, "hidden_dim", None)
        self.patch_dim: Tuple[int, int] = (0, 0)
        self.num_tokens: int = 0
        self._last_feature_hw: Optional[Tuple[int, int]] = None
        self._stem: Optional[nn.Module] = None
        if self.spec.head_type == "fc":
            self._stem = nn.Sequential(*(list(self.model.children())[:-2]))
        elif self.spec.head_type in {"convnext", "swin"} and hasattr(self.model, "features"):
            self._stem = self.model.features  # type: ignore[assignment]
        self._init_shape_metadata()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self._encode_tokens(x)

    def _init_shape_metadata(self) -> None:
        can_trust_grid = (
            self.spec.head_type == "vit"
            and self.patch_size is not None
            and self.hidden_dim is not None
        )
        if can_trust_grid:
            grid = self.image_size // self.patch_size
            self.patch_dim = (grid, grid)
            self.num_tokens = grid * grid
            return
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            tokens = self._encode_tokens(dummy)
            self.hidden_dim = tokens.shape[-1]
            self.num_tokens = tokens.shape[1]
            if self.patch_dim == (0, 0):
                side = int(math.sqrt(self.num_tokens)) if self.num_tokens > 0 else 0
                other = self.num_tokens // side if side else 0
                self.patch_dim = (side, other)
            if self.patch_size is None and self.patch_dim[0] > 0:
                self.patch_size = self.image_size // self.patch_dim[0]

    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.spec.head_type == "vit":
            return self._encode_vit_tokens(x)
        if self.spec.head_type == "fc":
            assert self._stem is not None
            features = self._stem(x)
            return self._flatten_features(features)
        if self.spec.head_type == "convnext":
            assert self._stem is not None
            features = self._stem(x)
            return self._flatten_features(features)
        if self.spec.head_type == "swin":
            if hasattr(self.model, "forward_features"):
                features = self.model.forward_features(x)  # type: ignore[attr-defined]
            else:
                assert self._stem is not None
                features = self._stem(x)
            if isinstance(features, tuple):
                features = features[0]
            if isinstance(features, dict):
                features = next(iter(features.values()))
            if features.ndim == 3:
                return features
            return self._flatten_features(features)
        raise ValueError(f"Unsupported head type: {self.spec.head_type}")

    def _encode_vit_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # Mirrors torchvision ViT forward but returns patch tokens only.
        processed = self.model._process_input(x)  # type: ignore[attr-defined]
        n = processed.shape[1]
        batch_class_token = self.model.class_token.expand(processed.shape[0], -1, -1)  # type: ignore[attr-defined]
        tokens = torch.cat((batch_class_token, processed), dim=1)
        if self.use_jepa_pos_with_backbone:
            return tokens[:, 1:, :]
        tokens = tokens + self.model.encoder.pos_embedding[:, : n + 1, :]  # type: ignore[attr-defined]
        tokens = self.model.encoder.dropout(tokens)  # type: ignore[attr-defined]
        tokens = self.model.encoder(tokens)  # type: ignore[attr-defined]
        tokens = self.model.encoder.ln(tokens)  # type: ignore[attr-defined]
        return tokens[:, 1:, :]

    def _flatten_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            return features.unsqueeze(1)
        if features.ndim != 4:
            raise ValueError("Expected convolutional features to be 4D")
        bsz, channels, h, w = features.shape
        self._last_feature_hw = (h, w)
        tokens = features.view(bsz, channels, h * w).transpose(1, 2)
        # Preserve the true patch grid for downstream consumers when available.
        self.patch_dim = (h, w)
        return tokens

class BackboneFeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, spec: BackboneSpec) -> None:
        super().__init__()
        self.model = model
        self.spec = spec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spec.head_type == "vit":
            processed = self.model._process_input(x)  # type: ignore[attr-defined]
            B, N, D = processed.shape
            # processed = processed.reshape(B, C, H * W).transpose(1, 2)  # B x N x C
            # n = processed.shape[1]
            #n = processed.shape[3]
            batch_class_token = self.model.class_token.expand(
                B, -1, -1
                #processed.shape, -1, -1
            )  # type: ignore[attr-defined]
            tokens = torch.cat((batch_class_token, processed), dim=1)
            tokens = tokens + self.model.encoder.pos_embedding[:, : N + 1, :]  # type: ignore[attr-defined]
            tokens = self.model.encoder.dropout(tokens)  # type: ignore[attr-defined]
            tokens = self.model.encoder(tokens)  # type: ignore[attr-defined]
            tokens = self.model.encoder.ln(tokens)  # type: ignore[attr-defined]
            cls = tokens[:, 0, :]  # B x D
            return cls

        if self.spec.head_type in {"convnext", "swin", "fc"}:
            feats = self.model(x)
            if isinstance(feats, tuple):
                feats = feats
            if isinstance(feats, dict):
                feats = next(iter(feats.values()))
            if feats.ndim == 4:
                feats = feats.mean(dim=[4][5]) # B x C x H x W -> B x C
            return feats

        raise ValueError(f"Unsupported head type for feature extractor: {self.spec.head_type}")

__all__ = [
    "adapt_config_for_backbone",
    "available_backbones",
    "build_backbone",
    "choose_num_heads",
    "get_backbone_spec",
    "resolve_preprocess_transforms",
    "BackboneFeatureExtractor",
]
