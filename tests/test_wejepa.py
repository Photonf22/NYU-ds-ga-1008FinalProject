"""Smoke tests for the wejepa package."""
from wejepa import IJEPADataset, IJepaConfig, create_pretraining_dataloader, default_config


def _fake_config() -> IJepaConfig:
    cfg = default_config()
    cfg.data.use_fake_data = True
    cfg.data.fake_data_size = 16
    cfg.data.train_batch_size = 4
    cfg.data.dataset_root = "./tests-data"
    return cfg


def test_dataset_returns_tensor(tmp_path):
    cfg = _fake_config()
    cfg.data.dataset_root = str(tmp_path)
    dataset = IJEPADataset(cfg, train=True)
    sample = dataset[0]
    assert sample.shape[-2:] == (cfg.data.image_size, cfg.data.image_size)


def test_dataloader_shapes(tmp_path):
    cfg = _fake_config()
    cfg.data.dataset_root = str(tmp_path)
    loader, _ = create_pretraining_dataloader(cfg)
    batch = next(iter(loader))
    assert batch.shape == (
        cfg.data.train_batch_size,
        3,
        cfg.data.image_size,
        cfg.data.image_size,
    )


def test_config_roundtrip():
    cfg = _fake_config()
    payload = cfg.to_dict()
    restored = IJepaConfig.from_dict(payload)
    assert restored.data.image_size == cfg.data.image_size
