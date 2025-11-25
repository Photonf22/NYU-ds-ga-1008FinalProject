import json
import json
from copy import deepcopy

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from wejepa import IJEPA_base, adapt_config_for_backbone, available_backbones, default_config
from wejepa.backbones import get_backbone_spec


@pytest.mark.parametrize("backbone", available_backbones())
def test_adapt_config_respects_backbone_defaults(backbone):
    cfg = adapt_config_for_backbone(default_config(), backbone)
    spec = get_backbone_spec(backbone)
    assert cfg.model.classification_backbone == backbone
    assert cfg.model.img_size == spec.default_image_size
    assert cfg.data.image_size == spec.default_image_size
    if spec.default_patch_size is not None:
        assert cfg.model.patch_size == spec.default_patch_size
    if spec.default_embed_dim is not None:
        assert cfg.model.embed_dim == spec.default_embed_dim
    assert cfg.model.embed_dim % cfg.model.num_heads == 0


@pytest.mark.parametrize("backbone", available_backbones())
def test_backbone_swaps_are_constructible(backbone):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    cfg = adapt_config_for_backbone(deepcopy(default_config()), backbone)
    cfg.model.classification_backbone = backbone

    model = IJEPA_base(
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        in_chans=cfg.model.in_chans,
        embed_dim=cfg.model.embed_dim,
        enc_depth=cfg.model.enc_depth,
        pred_depth=cfg.model.pred_depth,
        num_heads=cfg.model.num_heads,
        backbone=cfg.model.classification_backbone,
        pretrained=False,
    )

    dummy = torch.randn(2, cfg.model.in_chans, cfg.model.img_size, cfg.model.img_size)
    preds, targets = model(dummy)

    assert preds.shape[-1] == model.backbone.hidden_dim
    assert model.patch_dim == getattr(model.backbone, "patch_dim")
    assert model.num_tokens == getattr(model.backbone, "num_tokens")

    # ensure config can be serialized even after mutation
    json.dumps(cfg.to_dict())
