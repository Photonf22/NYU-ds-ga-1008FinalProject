"""
Fine-tuning utilities
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision

from ..config import IJepaConfig, default_config
from ..datasets.standard import (
    StandardImageDataset,
    build_standard_eval_transform as build_eval_transform,
    build_standard_train_transform as build_train_transform,
)
from ..model import IJEPA_base


@dataclass
class FinetuneConfig:

    ijepa: IJepaConfig = field(default_factory=default_config)
    batch_size: int = 128
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_classes: int = 100
    num_workers: int = 4
    checkpoint_path: Optional[str] = None
    debug: bool = False


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




def _build_dataset(cfg: IJepaConfig, train: bool, debug: bool = False):
    name = getattr(cfg.data, 'dataset_name', 'cifar100').lower()
    transform = build_train_transform(cfg) if train else build_eval_transform(cfg)
    if cfg.data.use_fake_data:
        if debug:
            print(f"[DEBUG] Using FakeData with size={cfg.data.fake_data_size} for {'train' if train else 'eval'}")
        return torchvision.datasets.FakeData(
            size=cfg.data.fake_data_size,
            image_size=(3, cfg.data.image_size, cfg.data.image_size),
            num_classes=max(cfg.data.fake_data_size, cfg.data.eval_batch_size),
            transform=transform,
        )
    dataset_root = Path(cfg.data.dataset_root)
    if getattr(cfg.data, "dataset_dir", None):
        dataset_root = dataset_root / cfg.data.dataset_dir
    split = "train" if train else "val"
    if debug:
        print(
            f"[DEBUG] Loading StandardImageDataset name={name} split={split} root={dataset_root}"
        )
    return StandardImageDataset(
        dataset_root,
        split,
        dataset_name=name,
        transform=transform,
        return_labels=True,
        fake_size=cfg.data.fake_data_size if cfg.data.use_fake_data else None,
        image_size=cfg.data.image_size,
        val_ratio=getattr(cfg.data, "val_split", 0.1),
    )



def create_finetune_dataloader(
    cfg: IJepaConfig, train: bool, batch_size: Optional[int] = None, debug: bool = False
) -> DataLoader:
    dataset = _build_dataset(cfg, train=train, debug=debug)
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size or (cfg.data.train_batch_size if train else cfg.data.eval_batch_size),
        shuffle=train,
        num_workers=cfg.data.num_workers,
        pin_memory=getattr(cfg.data, 'pin_memory', False),
        drop_last=train,
        persistent_workers=getattr(cfg.data, 'persistent_workers', False) and cfg.data.num_workers > 0,
    )
    if getattr(cfg.data, 'num_workers', 0) > 0 and hasattr(cfg.data, 'prefetch_factor'):
        kwargs["prefetch_factor"] = cfg.data.prefetch_factor
    if debug:
        print(
            f"[DEBUG] Created {'train' if train else 'eval'} dataloader with batch_size={kwargs['batch_size']} "
            f"workers={cfg.data.num_workers} shuffle={train} dataset_len={len(dataset)}"
        )
    loader = DataLoader(**kwargs)
    return loader


def load_backbone_from_checkpoint(
    checkpoint_path: str, cfg: Optional[IJepaConfig] = None, debug: bool = False
) -> IJEPA_base:
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
        M=None,
        layer_dropout=cfg.model.layer_dropout,
        backbone=cfg.model.classification_backbone,
        pretrained=cfg.model.classification_pretrained,
        debug=debug,
    )
    if debug:
        print(f"[DEBUG] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    student_state = checkpoint.get("student") or checkpoint
    module.student_encoder.load_state_dict(student_state)
    if "teacher" in checkpoint:
        module.teacher_encoder.load_state_dict(checkpoint["teacher"])
    if "predictor" in checkpoint:
        module.predictor.load_state_dict(checkpoint["predictor"])
    if debug:
        print(
            f"[DEBUG] Loaded student keys={len(student_state)} teacher_present={'teacher' in checkpoint} "
            f"predictor_present={'predictor' in checkpoint}"
        )
    module.set_mode("test")
    module.eval()
    return module


def _train_one_epoch(
    model: LinearProbe,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    debug: bool = False,
) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(loader):
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
        if debug and batch_idx == 0:
            print(
                f"[DEBUG] First train batch: images={tuple(images.shape)} logits={tuple(logits.shape)} "
                f"loss={loss.item():.4f}"
            )
    avg_loss = total_loss / max(1, total)
    accuracy = total_correct / max(1, total)
    return avg_loss, accuracy


def _evaluate(
    model: LinearProbe, loader: DataLoader, device: torch.device, debug: bool = False
) -> float:
    model.eval()
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
            if debug and batch_idx == 0:
                print(
                    f"[DEBUG] Eval batch: images={tuple(images.shape)} logits={tuple(logits.shape)}"
                )
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
        loss, acc = _train_one_epoch(
            model, train_loader, optimizer, device, debug=cfg.debug
        )
        val_acc = _evaluate(model, eval_loader, device, debug=cfg.debug)
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
    if ft_cfg.debug:
        print(f"[DEBUG] Using device {device} for fine-tuning")
    backbone = load_backbone_from_checkpoint(ft_cfg.checkpoint_path, cfg, debug=ft_cfg.debug)
    backbone.to(device)
    model = LinearProbe(backbone, ft_cfg.num_classes).to(device)
    optimizer = torch.optim.Adam(
        model.head.parameters(),
        lr=ft_cfg.learning_rate,
        weight_decay=ft_cfg.weight_decay,
    )
    train_loader = create_finetune_dataloader(
        cfg, train=True, batch_size=ft_cfg.batch_size, debug=ft_cfg.debug
    )
    eval_loader = create_finetune_dataloader(
        cfg, train=False, batch_size=ft_cfg.batch_size, debug=ft_cfg.debug
    )
    for epoch in range(ft_cfg.epochs):
        loss, acc = _train_one_epoch(
            model, train_loader, optimizer, device, debug=ft_cfg.debug
        )
        val_acc = _evaluate(model, eval_loader, device, debug=ft_cfg.debug)
        print(
            f"[Linear probe] Epoch {epoch + 1}/{ft_cfg.epochs} "
            f"| loss={loss:.4f} | train_acc={acc:.3f} | val_acc={val_acc:.3f}"
        )
    return model


def compare_pretrained_vs_scratch(ft_cfg: FinetuneConfig) -> FinetuneReport:

    cfg = ft_cfg.ijepa
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = create_finetune_dataloader(cfg, train=True, batch_size=ft_cfg.batch_size)
    eval_loader = create_finetune_dataloader(cfg, train=False, batch_size=ft_cfg.batch_size)

    # Pretrained probe
    pretrained_backbone = load_backbone_from_checkpoint(
        ft_cfg.checkpoint_path, cfg, debug=ft_cfg.debug
    )
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
        # Do not use mask config for finetuning
        M=None,
        layer_dropout=cfg.model.layer_dropout,
        backbone=cfg.model.classification_backbone,
        pretrained=cfg.model.classification_pretrained,
        debug=ft_cfg.debug,
    ).to(device)
    scratch_acc = _train_linear_probe_once(
        scratch_backbone, ft_cfg, device, train_loader, eval_loader
    )

    return FinetuneReport(pretrained_accuracy=pretrained_acc, scratch_accuracy=scratch_acc)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained WE-JEPA encoder")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file for fine-tuning")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a saved checkpoint (overrides config)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay (overrides config)")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes (overrides config)")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debugging output for fine-tuning setup and IO",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # Load config from file if provided
    config_data = {}
    if args.config is not None:
        if not os.path.isfile(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        with open(args.config, "r") as f:
            config_data = json.load(f)
    # Build FinetuneConfig, ensuring ijepa is loaded as IJepaConfig from config file if present
    if config_data:
        # If config_data is a flat dict, treat as IJepaConfig
        if "model" in config_data and "data" in config_data:
            ijepa_cfg = IJepaConfig.from_dict(config_data)
        # If config_data is a FinetuneConfig dict with 'ijepa' key
        elif "ijepa" in config_data:
            ijepa_cfg = IJepaConfig.from_dict(config_data["ijepa"])
        else:
            ijepa_cfg = default_config()
        ft_cfg = FinetuneConfig(ijepa=ijepa_cfg)
        # Set other FinetuneConfig fields if present in config_data
        for k, v in config_data.items():
            if hasattr(ft_cfg, k) and k != "ijepa":
                setattr(ft_cfg, k, v)
    else:
        ft_cfg = FinetuneConfig()
    # CLI args override config file
    if args.epochs is not None:
        ft_cfg.epochs = args.epochs
    if args.batch_size is not None:
        ft_cfg.batch_size = args.batch_size
    if args.lr is not None:
        ft_cfg.learning_rate = args.lr
    if args.weight_decay is not None:
        ft_cfg.weight_decay = args.weight_decay
    if args.num_classes is not None:
        ft_cfg.num_classes = args.num_classes
    if args.checkpoint is not None:
        ft_cfg.checkpoint_path = args.checkpoint
    if args.debug:
        ft_cfg.debug = True
    if ft_cfg.checkpoint_path is None:
        raise ValueError("A pretrained checkpoint path must be provided via --checkpoint or config file.")
    if ft_cfg.debug:
        print(f"[DEBUG] Fine-tune config: {ft_cfg}")
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
