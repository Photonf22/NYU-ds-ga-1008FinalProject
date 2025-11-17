"""CIFAR-100 download helpers and CLI entrypoints.

This module bundles the dataset bootstrap logic used by ``wejepa`` so that new
machines can be primed with the expected CIFAR-100 layout before running the
pretraining scripts.  It intentionally avoids any project-specific runtime
state (like accelerators or distributed processes) which allows the helper to
run from bare Python environments.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping

from torchvision.datasets import CIFAR100

_VALID_SPLITS = {"train", "test"}


def _normalize_root(dataset_root: str | Path) -> Path:
    """Return the absolute dataset root and make sure it exists."""

    root = Path(dataset_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def download_cifar100(
    dataset_root: str | Path,
    splits: Iterable[str] = ("train", "test"),
) -> Mapping[str, Path]:
    """Download CIFAR-100 splits into ``dataset_root`` if needed.

    Parameters
    ----------
    dataset_root:
        Directory where torchvision should store the CIFAR-100 archives.
    splits:
        Iterable containing any combination of ``"train"`` or ``"test"``.

    Returns
    -------
    Mapping[str, Path]
        Dictionary from split name to the directory that now contains the
        extracted data.
    """

    root = _normalize_root(dataset_root)
    downloaded: MutableMapping[str, Path] = {}
    for split in splits:
        if split not in _VALID_SPLITS:
            raise ValueError(f"Unknown split '{split}'. Expected one of {_VALID_SPLITS}.")
        train_flag = split == "train"
        CIFAR100(root=str(root), train=train_flag, download=True)
        downloaded[split] = root
    return downloaded


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the CIFAR-100 dataset so wejepa pretraining can reuse it "
            "without hitting the network."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path.cwd() / "data",
        help="Folder that should contain the torchvision CIFAR-100 download",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "test"),
        choices=sorted(_VALID_SPLITS),
        help="Which CIFAR-100 splits to materialize.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entrypoint used via ``python -m wejepa.data.download``."""

    args = _parse_args()
    downloads = download_cifar100(args.dataset_root, splits=args.splits)
    for split, root in downloads.items():
        print(f"Split '{split}' available under {root}")


if __name__ == "__main__":
    main()
