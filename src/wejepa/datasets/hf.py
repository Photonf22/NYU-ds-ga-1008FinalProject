"""Hugging Face dataset helpers for pretraining."""

from torch.utils.data import Dataset
from datasets import load_dataset

class HFDataset(Dataset):
    def __init__(self, cfg, split="train", transform=None):
        self.cfg = cfg
        self.transform = transform
        self.dataset = load_dataset(
            cfg.data.dataset_name,  # e.g., "tsbpp/fall2025_deeplearning"
            split=split,
            data_dir=cfg.data.dataset_root
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]["image"]  # column name may vary
        if self.transform:
            img = self.transform(img)
        return img

def build_train_transform(cfg):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.data.normalization_mean, cfg.data.normalization_std),
    ])

def build_eval_transform(cfg):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.data.normalization_mean, cfg.data.normalization_std),
    ])

def create_pretraining_dataloader(cfg, split="train"):
    from torch.utils.data import DataLoader
    transform = build_train_transform(cfg) if split == "train" else build_eval_transform(cfg)
    dataset = HFDataset(cfg, split=split, transform=transform)
    return DataLoader(
        dataset,
        batch_size=cfg.data.train_batch_size if split == "train" else cfg.data.eval_batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
        prefetch_factor=cfg.data.prefetch_factor,
    )