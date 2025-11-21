"""Tests for visualization and comparison helpers."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from wejepa import default_config
from wejepa.analysis import (
    extract_backbone_features,
    plot_tsne_embeddings,
    run_tsne_projection,
    visualize_attention_scores,
)
from wejepa.model import IJEPA_base
from wejepa.train import FinetuneConfig, compare_pretrained_vs_scratch


def _tiny_backbone() -> IJEPA_base:
    return IJEPA_base(
        img_size=32,
        patch_size=4,
        in_chans=3,
        embed_dim=64,
        enc_depth=2,
        pred_depth=1,
        num_heads=4,
        post_emb_norm=False,
        M=2,
    )


def test_visualize_attention_scores_returns_heatmap():
    backbone = _tiny_backbone()
    image = torch.randn(1, 3, 32, 32)
    fig, heatmap = visualize_attention_scores(backbone, image)
    assert heatmap.shape == backbone.patch_dim
    plt.close(fig)


def test_tsne_pipeline_produces_embeddings():
    backbone = _tiny_backbone()
    dataset = TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 3, (8,)))
    loader = DataLoader(dataset, batch_size=4)
    features, labels = extract_backbone_features(backbone, loader, max_batches=1)
    assert features.shape[0] == labels.shape[0]
    embedding = run_tsne_projection(features, perplexity=2.0, max_iter=250)
    fig = plot_tsne_embeddings(embedding, labels)
    plt.close(fig)


def test_compare_pretrained_vs_scratch_with_fake_data(tmp_path):
    cfg = default_config()
    cfg.data.dataset_root = str(tmp_path)
    cfg.data.use_fake_data = True
    cfg.data.fake_data_size = 4
    cfg.data.train_batch_size = 2
    cfg.data.eval_batch_size = 2
    cfg.data.num_workers = 0
    cfg.data.persistent_workers = False
    cfg.model.embed_dim = 64
    cfg.model.enc_depth = 2
    cfg.model.pred_depth = 1
    cfg.model.num_heads = 4
    cfg.mask.num_target_blocks = 2

    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_backbone = _tiny_backbone()
    torch.save({"student": checkpoint_backbone.student_encoder.state_dict()}, checkpoint_path)

    ft_cfg = FinetuneConfig(
        ijepa=cfg,
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        num_classes=5,
        checkpoint_path=str(checkpoint_path),
    )
    report = compare_pretrained_vs_scratch(ft_cfg)
    assert len(report.pretrained_accuracy) == 1
    assert len(report.scratch_accuracy) == 1
