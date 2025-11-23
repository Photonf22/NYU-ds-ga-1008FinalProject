# WeJEPA HPC pipeline client

`pipeline-py-client` is a terminal-friendly curses UI for running the WeJEPA
pipeline from a remote HPC login node.

## Installation

The client is a single executable script located in this folder. Make it
executable (if it is not already) and call it directly. The default behavior is
to launch the dialog-based UI.

```bash
chmod +x pipeline-py-client
./pipeline-py-client
```

Install the command to your 

> The UI requires the `dialog` binary (often preinstalled on HPC login nodes).
> If missing, install via your package manager (e.g., `apt-get install dialog`).
> All commands support `--dry-run` to print actions without executing them

## Interactive UI

Running `./pipeline-py-client` (or `./pipeline-py-client ui`) opens a menu with the core pipeline stages:

- **Install dependencies**: prompt for the pip command and execute
  `pip install --upgrade pip` followed by `pip install -e .[dev]`.
- **Download datasets**: choose datasets via checkboxes, enter the dataset root
  and splits, and build the `python -m wejepa.datasets.download` commands.
- **Launch pretraining**: choose a config, datasets, and extra arguments, then
  run `python -m wejepa.train.pretrain`. The flow can also generate and submit
  a Slurm script using the template under `slurms/`.
- **Generate Slurm**: wrap any arbitrary command in the Slurm template and
  optionally submit with `sbatch`.
- **Help**: open the README.md file inside a dialog textbox.

## Commands

### Install dependencies

```bash
./pipeline-py-client install
```

Runs the installation steps defined in the spec (`pip install --upgrade pip`
followed by `pip install -e .[dev]`). Use `--pip-command` to switch to `pip3`
or a virtual environment shim.

### Download datasets

```bash
./pipeline-py-client download \
  --dataset-root /path/to/datasets \
  --datasets cifar100 imagenet \
  --splits train test
```

Executes `python -m wejepa.datasets.download` once per dataset with the
provided splits. The dataset root must exist before running.

### Launch pretraining and optionally generate Slurm

```bash
./pipeline-py-client pretrain \
  --config configs/hf224_config.json \
  --datasets common_pool unsplash \
  --slurm-path my_pretrain.slurm \
  --job-name wejepa-pretrain \
  --gpus gpu:rtx8000:4 \
  --ext3-path /scratch/overlay.ext3 \
  --sif-path /scratch/env.sif \
  --submit \
  -- --batch-size 256 --epochs 100
```

This builds `python -m wejepa.train.pretrain` with the provided datasets and
extra arguments. When `--slurm-path` is provided a job file is generated from
`slurms/sbatch_pretrain.slurm`; `--submit` will call `sbatch` for you.

### Generate a Slurm script for any command

```bash
./pipeline-py-client slurm \
  --output download.slurm \
  --job-name wejepa-downloads \
  --ext3-path /scratch/overlay.ext3 \
  --sif-path /scratch/env.sif \
  -- -- python -m wejepa.datasets.download --dataset-root /path --dataset-name cifar100
```

Accepts the command to run after `--` and renders the submission script using
the same defaults as the pretraining helper.

### Inspect the bundled spec

```bash
./pipeline-py-client show-spec
```

Prints the `spec` document shipped with this directory for quick reference in a
remote shell.

## Notes

- Commands are executed exactly as printed; use `--dry-run` to rehearse them
  before launching on the cluster.
- The Slurm generator replaces the placeholders in `slurms/sbatch_pretrain.slurm`
  (job name, resources, overlay paths, and command body).
- If `sbatch` is not on your PATH, submission will fail with a clear error but
  the script will still be generated.

For a deeper dive into HPC and configuration knobs, see
[`docs/hpc_guide.md`](../docs/hpc_guide.md).
