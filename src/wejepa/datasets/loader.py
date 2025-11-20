"""Generic dataset loader for pretraining."""

def create_pretraining_dataloader(cfg, rank=0, world_size=1):
    dataset_name = getattr(cfg.data, "dataset_name", None)
    if dataset_name is None or dataset_name.lower() == "cifar100":
        from wejepa.datasets.cifar import create_pretraining_dataloader as cifar_loader
        return cifar_loader(cfg, rank=rank, world_size=world_size)
    else:
        from wejepa.datasets.hf import create_pretraining_dataloader as hf_loader
        return hf_loader(cfg, rank=rank, world_size=world_size, split="train")