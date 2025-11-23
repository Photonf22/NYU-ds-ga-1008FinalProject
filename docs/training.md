# Training WE-JEPA with the CLI

This guide shows how to launch pretraining runs with `python -m wejepa.train.pretrain` using different backbones and datasets.

## Common invocation

Run the entrypoint with a JSON config:

```bash
python -m wejepa.train.pretrain --config <path-to-config>.json
```

To inspect the default settings without training:

```bash
python -m wejepa.train.pretrain --print-config
```

Key knobs for backbone and initialization live under the `model` section of the config:

- `classification_backbone`: one of `swin_t`, `swin_s`, `convnext_tiny`, `convnext_small`, or `vit_b_16`.
- `classification_pretrained`: set to `false` to start from randomized weights (teacher weights are cloned from the randomly initialized student at startup).
- `img_size` / `patch_size` / `embed_dim` / `num_heads`: align these to the backbone’s native resolution and transformer shape. The helper `wejepa.backbones.adapt_config_for_backbone` can adjust a config for you inside a Python REPL or small script if you prefer automation.

## Example configs

### CIFAR-100 with ConvNeXt-Tiny (random init)

Save the following as `configs/cifar100_convnext_tiny.json`:

```json
{
  "data": {
    "dataset_root": "./data",
    "dataset_name": "cifar100",
    "image_size": 32,
    "train_batch_size": 256,
    "eval_batch_size": 512,
    "num_workers": 4,
    "pin_memory": true,
    "persistent_workers": true,
    "prefetch_factor": 2,
    "crop_scale": [0.6, 1.0],
    "color_jitter": 0.5,
    "use_color_distortion": true,
    "use_horizontal_flip": true,
    "normalization_mean": [0.5071, 0.4867, 0.4408],
    "normalization_std": [0.2675, 0.2565, 0.2761],
    "use_fake_data": false,
    "fake_data_size": 512
  },
  "mask": {
    "target_aspect_ratio": [0.75, 1.5],
    "target_scale": [0.15, 0.2],
    "context_aspect_ratio": 1.0,
    "context_scale": [0.85, 1.0],
    "num_target_blocks": 4
  },
  "model": {
    "img_size": 32,
    "patch_size": 4,
    "in_chans": 3,
    "embed_dim": 768,
    "enc_depth": 6,
    "pred_depth": 4,
    "num_heads": 12,
    "post_emb_norm": false,
    "layer_dropout": 0.0,
    "classification_backbone": "convnext_tiny",
    "classification_num_classes": 100,
    "classification_pretrained": false
  },
  "optimizer": {
    "epochs": 5,
    "warmup_epochs": 1,
    "base_learning_rate": 0.001,
    "start_learning_rate": 0.0001,
    "final_learning_rate": 1e-05,
    "weight_decay": 0.05,
    "final_weight_decay": 0.2,
    "betas": [0.9, 0.95],
    "eps": 1e-08,
    "grad_clip_norm": 1.0,
    "momentum_teacher": 0.996,
    "momentum_teacher_final": 1.0
  },
  "hardware": {
    "world_size": 1,
    "seed": 42,
    "mixed_precision": true,
    "compile_model": false,
    "log_every": 50,
    "checkpoint_every": 1,
    "output_dir": "./outputs/ijepa"
  }
}
```

Train with:

```bash
python -m wejepa.train.pretrain --config configs/cifar100_convnext_tiny.json
```

### Hugging Face 224×224 dataset with Swin-T (random init)

Start from the bundled 224×224 template and adjust for Swin-T without pretrained weights. Save as `configs/hf224_swin_t_random.json`:

```json
{
  "data": {
    "dataset_root": "./data",
    "dataset_name": "tsbpp___fall2025_deeplearning",
    "image_size": 224,
    "train_batch_size": 128,
    "eval_batch_size": 512,
    "num_workers": 16,
    "pin_memory": true,
    "persistent_workers": true,
    "prefetch_factor": 2,
    "crop_scale": [0.6, 1.0],
    "color_jitter": 0.5,
    "use_color_distortion": true,
    "use_horizontal_flip": true,
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    "use_fake_data": false,
    "fake_data_size": 512
  },
  "mask": {
    "target_aspect_ratio": [0.75, 1.5],
    "target_scale": [0.15, 0.2],
    "context_aspect_ratio": 1.0,
    "context_scale": [0.85, 1.0],
    "num_target_blocks": 4
  },
  "model": {
    "img_size": 224,
    "patch_size": 4,
    "in_chans": 3,
    "embed_dim": 768,
    "enc_depth": 6,
    "pred_depth": 4,
    "num_heads": 12,
    "post_emb_norm": false,
    "layer_dropout": 0.0,
    "classification_backbone": "swin_t",
    "classification_num_classes": 1000,
    "classification_pretrained": false
  },
  "optimizer": {
    "epochs": 5,
    "warmup_epochs": 1,
    "base_learning_rate": 0.001,
    "start_learning_rate": 0.0001,
    "final_learning_rate": 1e-05,
    "weight_decay": 0.05,
    "final_weight_decay": 0.2,
    "betas": [0.9, 0.95],
    "eps": 1e-08,
    "grad_clip_norm": 1.0,
    "momentum_teacher": 0.996,
    "momentum_teacher_final": 1.0
  },
  "hardware": {
    "world_size": 4,
    "seed": 42,
    "mixed_precision": true,
    "compile_model": false,
    "log_every": 50,
    "checkpoint_every": 1,
    "output_dir": "./outputs/ijepa"
  }
}
```

Launch training:

```bash
python -m wejepa.train.pretrain --config configs/hf224_swin_t_random.json
```

### Swapping to other backbones or datasets

- Replace `classification_backbone` with `swin_s` or `convnext_small` and align `embed_dim`/`num_heads` with the defaults from `wejepa.backbones._BACKBONES`.
- Use `classification_num_classes` that matches your dataset (e.g., 100 for CIFAR-100, 1 000 for ImageNet-style datasets).
- For quick experiments, toggle `hardware.world_size` to match the number of GPUs and `optimizer.epochs` to shorten or extend training.

With these configs, both student and teacher encoders start from randomized weights and train entirely within the provided CLI entrypoint.
