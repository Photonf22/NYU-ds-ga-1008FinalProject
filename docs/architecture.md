# WE-JEPA architecture guide

This document summarizes the design of the WE-JEPA implementation shipped in this
repository. It complements the top-level README by explaining the neural network
blocks, configuration options, and how to run the main entry points.

## Model overview

The `wejepa.model.IJEPA_base` module implements a compact Joint-Embedding
Predictive Architecture:

* **Patch embedding** – Images are split into non-overlapping patches using a
  strided convolution (`PatchEmbed`). When `model.classification_backbone` is
  set, the class wraps a torchvision classifier and converts its intermediate
  feature maps into patch tokens instead.
* **Positional encoding** – A learnable positional embedding is added to the
  patch tokens. When a backbone changes the token grid (e.g., switching from
  32×32 to 224×224 inputs) the embeddings are resized on the fly to match the
  new sequence length.
* **Student/teacher encoders** – Transformer encoders process the context tokens.
  The teacher is an exponential moving average copy of the student to stabilize
  training.
* **Predictor** – A lightweight decoder that consumes the encoded context tokens
  plus masked target tokens to predict the teacher outputs for the masked
  regions.
* **Attention recording** – `IJEPA_base.enable_attention_recording(True)`
  attaches hooks to every encoder layer so that visualization utilities can
  inspect the per-head attention weights after a forward pass.

## Training and fine-tuning entry points

* **Pretraining** lives in `wejepa.train.pretrain.launch_pretraining`. The
  function builds the JEPA model, data loaders, cosine schedules, and supports
  optional distributed training via `torch.distributed`.
* **Linear probing and comparisons** are handled in
  `wejepa.train.finetune.train_linear_probe` and
  `compare_pretrained_vs_scratch`. These routines freeze the JEPA encoder,
  attach a linear head, and report validation accuracy across epochs. They are
  designed to run against `FakeData` when `cfg.data.use_fake_data=True` so smoke
  tests finish in seconds.

## Configuration

Hyperparameters are grouped in `IJepaConfig` (see `src/wejepa/config.py`). The
key sections are:

* `data`: dataset root, batch sizes, augmentations, and whether to use
  `torchvision.datasets.FakeData` for quick iterations.
* `mask`: controls the masking strategy (target/context aspect ratios and scales
  plus the number of target blocks per sample).
* `model`: image/patch sizes, embedding dimensions, transformer depth, number of
  heads, and optional classification backbone integration.
* `optimizer`: AdamW settings and cosine schedules for learning rate, weight
  decay, and teacher momentum.
* `hardware`: seeds, mixed-precision toggle, checkpoint cadence, and output
  directory.

Use `default_config()` for 32×32 CIFAR-style experiments or
`hf224_config()`/`adapt_config_for_backbone()` for ImageNet-scale backbones. Any
configuration can be serialized to/from JSON via `IJepaConfig.to_dict()` and
`IJepaConfig.from_dict()`.

## Usage tips

* Start with `python -m wejepa.train.pretrain --print-config` to verify that
  dependencies import cleanly and serialization works.
* When experimenting with different backbones, call
  `adapt_config_for_backbone(default_config(), "vit_b_16")` to align the image
  size, patch size, and attention head count with the chosen architecture.
* For quick evaluations or CI, set `cfg.data.use_fake_data = True` and
  `cfg.data.fake_data_size` to a small number. All provided tests and helpers are
  compatible with fake data.
* Visualize attention with
  `wejepa.analysis.visualization.visualize_attention_scores`, which records
  encoder attention maps and produces a heatmap over the patch grid.

These conventions aim to make the codebase easy to extend while keeping the
training pipelines fully reproducible.
