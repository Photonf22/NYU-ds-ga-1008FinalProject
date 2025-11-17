# Dataset setup

All helpers in
`wejepa.data` assume the dataset lives under `cfg.data.dataset_root` (defaults to
`<repo>/data`).

## One-time download

Use the built-in CLI to materialize the train and test archives:

```bash
python -m wejepa.datasets.download --dataset-root ./data
```

The command only downloads missing files, so it is safe to re-run when syncing a
shared filesystem.  Pass `--splits train` to fetch just the training archive or
`--splits test` to refresh the held-out split.

## Validating a machine

`python -m wejepa.train.pretrain --print-config` is a lightweight sanity check
that confirms the training entrypoint can be imported and the default
configuration points at the expected dataset root.  When running on new
hardware, prefer the following order:

1. `python -m wejepa.datasets.download --dataset-root <path>`
2. `python -m wejepa.train.pretrain --print-config`
3. `python -m wejepa.train.pretrain --config configs/cifar100_base.json`

This ensures the storage is writable, dataset exists, and the job can start
without failing late because of missing files.
