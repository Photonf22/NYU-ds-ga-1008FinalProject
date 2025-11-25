# WE-JEPA

## Important Dates
* Initial test + Kaggle platform release: November 18, 2025.
* Final public test release: November 25, 2025.
* Final submission deadline: December 2, 2025 (11:59pm local time).
* Report: Due during the exam period. 4 pages (excluding references), with citations, using the **CVPR template.**

Reference implementation of a lightweight Image-based Joint-Embedding Predictive Architecture (I-JEPA) style encoder that we
use for the DS-GA-1008 final project.

* configuration objects that capture every important hyper-parameter (`wejepa.config`),
* a dataset based unlabeled pretraining pipeline with distributed support (`wejepa.train.pretrain`),
* fine-tuning helpers and linear-probe utilities (`wejepa.train.finetune`), and
* smoke tests plus documentation to make contributions safer.

The instructions below assume you are inside the project root.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

The editable install makes the `wejepa` package available everywhere and also registers
the extra developer dependencies (pytest, ruff, mypy, ...).

## Retaining / Pretraining the encoder

The main entrypoint lives in `wejepa.train.pretrain`.  You can explore the default
configuration without touching any data via the `--print-config` flag:

```bash
python -m wejepa.train.pretrain --print-config
```

To launch a real run you only need to make sure the dataset root exists and then call
`launch_pretraining` through the CLI.  The example below downloads data into
`/path/to/datasets` and runs the default single GPU/CPU training loop using fake data for
quick smoke testing.

```bash
python -m wejepa.train.pretrain \
  --config configs/cifar100_base.json  # optional JSON version of IJepaConfig
```

You can also drive training from Python:

```python
from wejepa import default_config, launch_pretraining

cfg = default_config()
cfg.data.dataset_root = "/path/to/datasets"
launch_pretraining(cfg)
```

Checkpoints (student, teacher and predictor weights) are stored in
`cfg.hardware.output_dir` after every epoch.

For a deeper dive into the architecture and configuration knobs, see
[`docs/architecture.md`](docs/architecture.md).

## Fine-tuning / linear probing

Fine-tuning utilities live next to the pretraining code.  The module exposes a
`FinetuneConfig` dataclass, a `LinearProbe` model and the `train_linear_probe` helper
which trains a frozen backbone plus linear head on labeled data.

```bash
python -m wejepa.train.finetune \
  --checkpoint outputs/ijepa/ijepa_epoch_0005.pt \
  --epochs 10 \
  --batch-size 256 \
  --lr 3e-4 \
  --num-classes 100
```

Or from Python for more control:

```python
from wejepa.train import FinetuneConfig, train_linear_probe

ft_cfg = FinetuneConfig(
    checkpoint_path="outputs/ijepa/ijepa_epoch_0005.pt",
    epochs=5,
    batch_size=128,
    learning_rate=1e-3,
)
train_linear_probe(ft_cfg)
```

The helper automatically downloads dataset labels if needed and reports
training/validation accuracy every epoch.

## Running inference

Use `load_backbone_from_checkpoint` to hydrate the encoder weights and then feed images
through the model in `test` mode.

```python
import torch
from torchvision import transforms
from PIL import Image

from wejepa.train import load_backbone_from_checkpoint
from wejepa import default_config

cfg = default_config()
backbone = load_backbone_from_checkpoint("outputs/ijepa/ijepa_epoch_0005.pt", cfg)
backbone.eval()

transform = transforms.Compose([
    transforms.Resize(cfg.data.image_size),
    transforms.ToTensor(),
    transforms.Normalize(cfg.data.normalization_mean, cfg.data.normalization_std),
])

image = transform(Image.open("/path/to/sample.jpg")).unsqueeze(0)
with torch.no_grad():
    tokens = backbone(image)
    pooled = tokens.mean(dim=1)  # embeddings for downstream heads
```

The returned tensor contains a per-sample embedding that can be consumed by any custom
head or downstream task.

## Tests and validation

Automated checks live in `tests/`.  Run them locally before pushing changes:

```bash
pytest -q
```

The dataset/dataloader smoke tests use `torchvision.datasets.FakeData`, so they pass even
without downloading datasets and finish in a couple of seconds.  For configuration
sanity checks you can also run:

```bash
python -m wejepa.train.pretrain --print-config
```

which validates that the training entrypoint can be imported and that serialization of
`IJepaConfig` works.

