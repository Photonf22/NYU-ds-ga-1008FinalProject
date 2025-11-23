"""Fine-tuning utilities for WE-JEPA backbones."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision

from ..config import IJepaConfig, default_config
from ..datasets.cifar import build_eval_transform, build_train_transform
from ..model import IJEPA_base


@dataclass
class FinetuneConfig:
    """Configuration for running the linear-probe fine-tuning loop."""

    ijepa: IJepaConfig = field(default_factory=default_config)
    batch_size: int = 128
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_classes: int = 100
    num_workers: int = 4
    checkpoint_path: Optional[str] = None


@dataclass
class FinetuneReport:
    """Accuracy trace comparing pretrained and scratch linear probes."""

    pretrained_accuracy: List[float]
    scratch_accuracy: List[float]


class LinearProbe(nn.Module):
    """Average pooled linear probe on top of the JEPA student encoder."""

    def __init__(self, backbone: IJEPA_base, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone.set_mode("test")
        for param in self.backbone.parameters():
            param.requires_grad = False
        embed_dim = self.backbone.pos_embedding.shape[-1]
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(x)
        pooled = tokens.mean(dim=1)
        return self.head(pooled)


def _cifar_dataset(cfg: IJepaConfig, train: bool) -> torchvision.datasets.CIFAR100:
    transform = build_train_transform(cfg) if train else build_eval_transform(cfg)
    if cfg.data.use_fake_data:
        return torchvision.datasets.FakeData(
            size=cfg.data.fake_data_size,
            image_size=(3, cfg.data.image_size, cfg.data.image_size),
            num_classes=max(cfg.data.fake_data_size, cfg.data.eval_batch_size),
            transform=transform,
        )
    return torchvision.datasets.CIFAR100(
        root=cfg.data.dataset_root,
        train=train,
        transform=transform,
        download=train,
    )
    return dataset


def create_finetune_dataloader(
    cfg: IJepaConfig, train: bool, batch_size: Optional[int] = None
) -> DataLoader:
    dataset = _cifar_dataset(cfg, train=train)
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size or (cfg.data.train_batch_size if train else cfg.data.eval_batch_size),
        shuffle=train,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=train,
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
    )
    if cfg.data.num_workers > 0:
        kwargs["prefetch_factor"] = cfg.data.prefetch_factor
    loader = DataLoader(**kwargs)
    return loader


def load_backbone_from_checkpoint(checkpoint_path: str, cfg: Optional[IJepaConfig] = None) -> IJEPA_base:
    cfg = cfg or default_config()
    module = IJEPA_base(
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        in_chans=cfg.model.in_chans,
        embed_dim=cfg.model.embed_dim,
        enc_depth=cfg.model.enc_depth,
        pred_depth=cfg.model.pred_depth,
        num_heads=cfg.model.num_heads,
        post_emb_norm=cfg.model.post_emb_norm,
        M=cfg.mask.num_target_blocks,
        layer_dropout=cfg.model.layer_dropout,
        backbone=cfg.model.classification_backbone,
        pretrained=cfg.model.classification_pretrained,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    student_state = checkpoint.get("student") or checkpoint
    module.student_encoder.load_state_dict(student_state)
    if "teacher" in checkpoint:
        module.teacher_encoder.load_state_dict(checkpoint["teacher"])
    if "predictor" in checkpoint:
        module.predictor.load_state_dict(checkpoint["predictor"])
    module.set_mode("test")
    module.eval()
    return module


def _train_one_epoch(
    model: LinearProbe,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    avg_loss = total_loss / max(1, total)
    accuracy = total_correct / max(1, total)
    return avg_loss, accuracy


def _evaluate(model: LinearProbe, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
    return total_correct / max(1, total)


def _train_linear_probe_once(
    backbone: IJEPA_base,
    cfg: FinetuneConfig,
    device: torch.device,
    train_loader: DataLoader,
    eval_loader: DataLoader,
) -> List[float]:
    model = LinearProbe(backbone, cfg.num_classes).to(device)
    optimizer = torch.optim.Adam(
        model.head.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    accuracies: List[float] = []
    for epoch in range(cfg.epochs):
        loss, acc = _train_one_epoch(model, train_loader, optimizer, device)
        val_acc = _evaluate(model, eval_loader, device)
        accuracies.append(val_acc)
        print(
            f"[Linear probe] Epoch {epoch + 1}/{cfg.epochs} "
            f"| loss={loss:.4f} | train_acc={acc:.3f} | val_acc={val_acc:.3f}"
        )
    return accuracies


def train_linear_probe(ft_cfg: Optional[FinetuneConfig] = None) -> LinearProbe:
    ft_cfg = ft_cfg or FinetuneConfig()
    cfg = ft_cfg.ijepa
    if ft_cfg.checkpoint_path is None:
        raise ValueError("A pretrained checkpoint path must be provided for fine-tuning.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_backbone_from_checkpoint(ft_cfg.checkpoint_path, cfg)
    backbone.to(device)
    model = LinearProbe(backbone, ft_cfg.num_classes).to(device)
    optimizer = torch.optim.Adam(
        model.head.parameters(),
        lr=ft_cfg.learning_rate,
        weight_decay=ft_cfg.weight_decay,
    )
    train_loader = create_finetune_dataloader(cfg, train=True, batch_size=ft_cfg.batch_size)
    eval_loader = create_finetune_dataloader(
        cfg, train=False, batch_size=ft_cfg.batch_size
    )
    for epoch in range(ft_cfg.epochs):
        loss, acc = _train_one_epoch(model, train_loader, optimizer, device)
        val_acc = _evaluate(model, eval_loader, device)
        print(
            f"[Linear probe] Epoch {epoch + 1}/{ft_cfg.epochs} "
            f"| loss={loss:.4f} | train_acc={acc:.3f} | val_acc={val_acc:.3f}"
        )
    return model


def compare_pretrained_vs_scratch(ft_cfg: FinetuneConfig) -> FinetuneReport:
    """Train probes on a pretrained backbone vs. a fresh one for quick validation."""

    cfg = ft_cfg.ijepa
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = create_finetune_dataloader(cfg, train=True, batch_size=ft_cfg.batch_size)
    eval_loader = create_finetune_dataloader(cfg, train=False, batch_size=ft_cfg.batch_size)

    # Pretrained probe
    pretrained_backbone = load_backbone_from_checkpoint(ft_cfg.checkpoint_path, cfg)
    pretrained_backbone.to(device)
    pretrained_acc = _train_linear_probe_once(
        pretrained_backbone, ft_cfg, device, train_loader, eval_loader
    )

    # Scratch probe
    scratch_backbone = IJEPA_base(
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        in_chans=cfg.model.in_chans,
        embed_dim=cfg.model.embed_dim,
        enc_depth=cfg.model.enc_depth,
        pred_depth=cfg.model.pred_depth,
        num_heads=cfg.model.num_heads,
        post_emb_norm=cfg.model.post_emb_norm,
        M=cfg.mask.num_target_blocks,
        layer_dropout=cfg.model.layer_dropout,
        backbone=cfg.model.classification_backbone,
        pretrained=cfg.model.classification_pretrained,
    ).to(device)
    scratch_acc = _train_linear_probe_once(
        scratch_backbone, ft_cfg, device, train_loader, eval_loader
    )

    return FinetuneReport(pretrained_accuracy=pretrained_acc, scratch_accuracy=scratch_acc)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained WE-JEPA encoder")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved checkpoint")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-classes", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ft_cfg = FinetuneConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_classes=args.num_classes,
        checkpoint_path=args.checkpoint,
    )
    train_linear_probe(ft_cfg)


__all__ = [
    "FinetuneConfig",
    "FinetuneReport",
    "LinearProbe",
    "create_finetune_dataloader",
    "compare_pretrained_vs_scratch",
    "load_backbone_from_checkpoint",
    "train_linear_probe",
]


if __name__ == "__main__":
    main()
