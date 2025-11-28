"""Utility helpers for fine-tuning a pretrained WE-JEPA encoder."""
from __future__ import annotations

from tqdm import tqdm
import json
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import sys
import os
from torchvision import transforms 
from PIL import Image
from pathlib import Path
from wejepa.backbones import adapt_config_for_backbone, available_backbones

SCRIPT_DIR = "~/code/dl_project1_copy_a_copy/NYU-ds-ga-1008FinalProject/src/wejepa"
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
#sys.path.append(str(Path(file).parent.parent))
from config import IJepaConfig, default_config
#from ..datasets.cifar import build_eval_transform, build_train_transform
from model import IJEPA_base
transform_to_tensor = transforms.ToTensor()

def collate_fn(batch):
    """Custom collate function to handle PIL images"""
    if len(batch[0]) == 3:  # train/val (image, label, filename)
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        filenames = [item[2] for item in batch]
        return images, labels, filenames
    else:  # test (image, filename)
        images = [item[0] for item in batch]
        filenames = [item[1] for item in batch]
        return images, filenames
class ImageDataset(Dataset):
    """Simple dataset for loading images"""

    def __init__(self, image_dir, image_list, labels=None, resolution=224):
        """
        Args:
            image_dir: Directory containing images
            image_list: List of image filenames
            labels: List of labels (optional, for train/val)
            resolution: Image resolution (96 for competition, 224 for DINO baseline)
        """
        self.image_dir = Path(image_dir)
        self.image_list = image_list
        self.labels = labels
        self.resolution = resolution
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = self.image_dir / img_name

        # Load and resize image
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.resolution, self.resolution), Image.BILINEAR)

        if self.labels is not None:
            return image, self.labels[idx], img_name
        return image, img_name

@dataclass
class FinetuneConfig:
    """Configuration for running the linear-probe fine-tuning loop."""

    ijepa: IJepaConfig = field(default_factory=default_config)
    config: Optional[str] = None
    batch_size: int = 128
    epochs: int = 6
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_classes: int = 100
    num_workers: int = 4
    checkpoint_path: Optional[str] = None
    type_of_backbone: Optional[str] = None


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

    def extract_batch_features(self, images):
        inputs = self.processor(images=images, return_tesnors="pt")
        input = {k: v.to(self.device) for k,v in inputs.item()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # cls token features
        cls_features = outputs.last_hidden_state[:,0]
        return cls_features.cpu().numpy()

    def extract_features(self, image):
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features
        # Please keep the model backbone frozen for the competition!
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use CLS token features
        cls_features = outputs.last_hidden_state[:, 0]  # Shape: (1, feature_dim)

        # Alternative: use mean of patch features

        # patch_features = outputs.last_hidden_state[:, 1:]
        # features = patch_features.mean(dim=1)
        return cls_features.cpu().numpy()[0]

def _cifar_dataset(cfg: IJepaConfig, train: bool) -> torchvision.datasets.CIFAR100:
    transform = build_train_transform(cfg) if train else build_eval_transform(cfg)
    dataset = torchvision.datasets.CIFAR100(
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
        batch_size=batch_size or cfg.data.train_batch_size,
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

def create_kaggle_dataloader(cfg: IJepaConfig, ImgDataSet: ImageDataset,train: bool, batch_size: Optional[int] = None) -> DataLoader:
        #dataset = _cifar_dataset(cfg, train=train)
        kwargs = dict(dataset=ImgDataSet,
                      batch_size=batch_size or cfg.data.train_batch_size,
                      shuffle=train,num_workers=cfg.data.num_workers,
                      pin_memory=cfg.data.pin_memory, 
                      drop_last=train,
                      persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
                      collate_fn=collate_fn
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
    #print("Loaded model: ", module)
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
    split_name='train'
    
    #for images, labels, filename in loader:
    for batch in tqdm(loader, desc=f"{split_name} features"):
        images, labels, filenames = batch
        #[print(label) for label in labels]
        
        #exit()
        tensor_list_img = [transform_to_tensor(img) for img in images]
        #tensor_list_label = [transform_to_tensor(label) for label in labels]
        #print(tensor_list_img)
        #exit()
        labels = torch.tensor(labels).to(device)
        images = torch.stack(tensor_list_img).to(device)
        
        #labels = torch.stack(tensor_list_label)
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


def train_linear_probe(ft_cfg: Optional[FinetuneConfig] = None, ImgDataSetTrain: Optional[ImageDataset] = None, ImgDataSetTest: Optional[ImageDataset] = None) -> LinearProbe:
    ft_cfg = ft_cfg or FinetuneConfig()
    
    #cfg = ft_cfg.ijepa
    #cfg = adapt_config_for_backbone(default_config(), type_of_backbone)
    
    if ft_cfg.config:
        print("adding ft_cfg config to loaded model")
        cfg_dict = json.loads(Path(ft_cfg.config).read_text())
        cfg_ijepa = IJepaConfig.from_dict(cfg_dict)
        cfg = adapt_config_for_backbone(cfg_ijepa, ft_cfg.type_of_backbone)
    else: 
        cfg = adapt_config_for_backbone(default_config(), ft_cfg.type_of_backbone)

    print("Configuration: ", cfg)
    if ft_cfg.checkpoint_path is None:
        raise ValueError("A pretrained checkpoint path must be provided for fine-tuning.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading backbone from checkpoint!")
    backbone = load_backbone_from_checkpoint(ft_cfg.checkpoint_path, cfg)
    backbone.to(device)
    model = LinearProbe(backbone, ft_cfg.num_classes).to(device)
    optimizer = torch.optim.Adam(
        model.head.parameters(),
        lr=ft_cfg.learning_rate,
        weight_decay=ft_cfg.weight_decay,
    )
    train_loader = create_kaggle_dataloader(cfg, ImgDataSetTrain, train=True, batch_size=ft_cfg.batch_size)
    if ImgDataSetTest != None:
        eval_loader = create_kaggle_dataloader(cfg, ImgDataSetTest, train=False, batch_size=ft_cfg.batch_size)
    #train_loader = create_finetune_dataloader(cfg, train=True, batch_size=ft_cfg.batch_size)
    #eval_loader = create_finetune_dataloader(
    #    cfg, train=False, batch_size=ft_cfg.batch_size
    #)
    for epoch in range(ft_cfg.epochs):
        loss, acc = _train_one_epoch(model, train_loader, optimizer, device)
        if ImgDataSetTest !=None:
            val_acc = _evaluate(model, eval_loader, device)
            print(
                f"[Linear probe] Epoch {epoch + 1}/{ft_cfg.epochs} "
                f"| loss={loss:.4f} | train_acc={acc:.3f} | val_acc={val_acc:.3f}"
            )
        else:
            print(
                f"[Linear probe] Epoch {epoch + 1}/{ft_cfg.epochs} "
                f"| loss={loss:.4f} | train_acc={acc:.3f}"
            )
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained WE-JEPA encoder")
    parser.add_argument("--data_dir",type=str,required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved checkpoint")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--model_name", type=str, default='wejepa')
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--config", type=str)
    parser.add_argument("--type_of_backbone",type=str,required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print("Fine Tune Create")
    ft_cfg = FinetuneConfig(
        #data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_classes=args.num_classes,
        checkpoint_path=args.checkpoint,
        config=args.config,
        type_of_backbone=args.type_of_backbone
    )
    

    data_dir = Path(args.data_dir)
    # Load CSV files
    print("\nLoading dataset metadata...")
    train_df = pd.read_csv(data_dir / 'train_labels.csv')
    val_df = pd.read_csv(data_dir / 'val_labels.csv')
    test_df = pd.read_csv(data_dir / 'test_images.csv')

    print(f"  Train: {len(train_df)} images")
    print(f"  Val: {len(val_df)} images")
    print(f"  Test: {len(test_df)} images")
    print(f"  Classes: {train_df['class_id'].nunique()}")

    train_dataset = ImageDataset(
            data_dir / 'train',
            train_df['filename'].tolist(),
            train_df['class_id'].tolist(),
            resolution=args.resolution
    )
    print("Training dataset setup complete")

    #val_dataset = ImageDataset(
    #        data_dir / 'val',
    #        val_df['filename'].tolist(),
    #        val_df['class_id'].tolist(),
    #        resolution=args.resolution
    #)
    #  ['convnext_small', 'convnext_tiny', 'swin_s', 'swin_t', 'vit_b_16']
    train_linear_probe(ft_cfg, train_dataset)


__all__ = [
    "FinetuneConfig",
    "LinearProbe",
    #"create_finetune_dataloader",
    "create_kaggle_dataloader",
    "load_backbone_from_checkpoint",
    "train_linear_probe",
]


if __name__ == "__main__":
    main()
