"""Exploratory analysis helpers for WE-JEPA models."""
from .visualization import (
    extract_backbone_features,
    plot_tsne_embeddings,
    run_tsne_projection,
    visualize_attention_scores,
)

__all__ = [
    "extract_backbone_features",
    "run_tsne_projection",
    "plot_tsne_embeddings",
    "visualize_attention_scores",
]
