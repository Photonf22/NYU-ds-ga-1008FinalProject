"""Visualization helpers for WE-JEPA backbones."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from ..model import IJEPA_base


def visualize_attention_scores(
    backbone: IJEPA_base,
    image: torch.Tensor,
    layer_index: int = -1,
    cmap: str = "magma",
) -> Tuple[plt.Figure, np.ndarray]:
    """Visualize attention scores for a single image.

    Parameters
    ----------
    backbone:
        Frozen WE-JEPA backbone in evaluation mode.
    image:
        Tensor of shape ``(1, 3, H, W)`` that has already been normalized with the
        training statistics.
    layer_index:
        Which transformer layer to visualize. Defaults to the last encoder layer.
    cmap:
        Matplotlib colormap to use for the heatmap.

    Returns
    -------
    fig, heatmap:
        A matplotlib figure plus the normalized attention heatmap with shape matching
        the patch grid.
    """

    backbone.set_mode("test")
    backbone.enable_attention_recording(True)
    device = next(backbone.parameters()).device
    with torch.no_grad():
        _ = backbone(image.to(device))
    attention_maps = backbone.get_recorded_attentions()
    backbone.enable_attention_recording(False)
    if not attention_maps:
        raise RuntimeError("Backbone did not record any attention maps.")
    selected = attention_maps[layer_index]
    averaged = selected.mean(dim=1)[0].mean(dim=0)  # (seq_len,)
    heatmap = averaged.reshape(backbone.patch_dim).cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)

    # Prepare the original image for visualization by min/max scaling.
    image_np = image[0].detach().cpu()
    image_np = image_np - image_np.min()
    if image_np.max() > 0:
        image_np = image_np / image_np.max()
    image_np = image_np.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Input image")
    axes[0].axis("off")
    im = axes[1].imshow(heatmap, cmap=cmap)
    axes[1].set_title("Attention heatmap")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    fig.tight_layout()
    return fig, heatmap


def extract_backbone_features(
    backbone: IJEPA_base,
    dataloader: DataLoader,
    max_batches: Optional[int] = 4,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a backbone over a dataloader and collect pooled features."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    backbone.set_mode("test")
    backbone.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch_idx, (images, target) in enumerate(dataloader):
            images = images.to(device)
            pooled = backbone(images).mean(dim=1).cpu()
            features.append(pooled)
            labels.append(target.cpu())
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break
    if not features:
        raise ValueError("Dataloader did not yield any batches.")
    feature_array = torch.cat(features, dim=0).numpy()
    label_array = torch.cat(labels, dim=0).numpy()
    return feature_array, label_array


def run_tsne_projection(
    features: np.ndarray,
    perplexity: float = 30.0,
    random_state: int = 0,
    max_iter: int = 1000,
    **kwargs,
) -> np.ndarray:
    """Project feature vectors into 2-D using T-SNE."""

    n_samples = max(1, features.shape[0])
    adjusted_perplexity = min(perplexity, max(1.0, n_samples - 1))
    tsne = TSNE(
        n_components=2,
        init="random",
        perplexity=adjusted_perplexity,
        random_state=random_state,
        max_iter=max_iter,
        **kwargs,
    )
    return tsne.fit_transform(features)


def plot_tsne_embeddings(
    embedding: np.ndarray,
    labels: Optional[np.ndarray] = None,
    palette: str = "tab10",
) -> plt.Figure:
    """Create a scatter plot from 2-D embeddings."""

    fig, ax = plt.subplots(figsize=(6, 5))
    if labels is None:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=15, alpha=0.8)
    else:
        labels = np.asarray(labels)
        unique = np.unique(labels)
        cmap = plt.get_cmap(palette, len(unique))
        for idx, cls in enumerate(unique):
            mask = labels == cls
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=20,
                alpha=0.8,
                color=cmap(idx),
                label=str(cls),
            )
        ax.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel("TSNE-1")
    ax.set_ylabel("TSNE-2")
    ax.set_title("T-SNE projection of JEPA features")
    fig.tight_layout()
    return fig
