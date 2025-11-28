"""
Data download helpers and CLI entrypoints.
"""
from __future__ import annotations

import argparse
import zipfile, tarfile
import tqdm
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Optional

from torchvision.datasets import CIFAR100
import huggingface_hub
import datasets
import urllib.request
import tarfile
import shutil

# For CUB-200 postprocessing
from .cub200 import CUB200Dataset
from .standard import ensure_standard_metadata

_VALID_SPLITS = {"train", "test"}


def _normalize_root(dataset_root: str | Path) -> Path:
    """Return the absolute dataset root and make sure it exists."""

    root = Path(dataset_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root

def download(
    dataset_root: str | Path,
    dataset_name: str = "cifar100",
    snapshot_download: Optional[bool] = False,
    splits: Iterable[str] = ("train", "test"),
    debug: bool = False,
) -> Mapping[str, Path]:
    """
    Parameters
    ----------
    dataset_root:
        Directory where we store the archives.
    splits:
        Iterable containing any combination of ``"train"`` or ``"test"``.

    Returns
    -------
    Mapping[str, Path]
    """

    root = _normalize_root(dataset_root)
    downloads_dir = root / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    if debug:
        print(
            f"[DEBUG] Preparing download for dataset={dataset_name} splits={tuple(splits)} root={root} downloads_dir={downloads_dir}"
        )
    downloaded: MutableMapping[str, Path] = {}
    def is_hf_format(directory: Path):
        for ext in [".csv", ".json", ".parquet"]:
            if any(directory.glob(f"*{ext}")):
                return True
        if any(p.is_dir() for p in directory.iterdir()):
            return True
        return False
    dataset_key = str(dataset_name).lower().replace("_", "").replace("-", "")

    def extract_archives(directory: Path):
        zip_files = list(directory.rglob("*.zip"))
        tar_files = list(directory.rglob("*.tar")) + list(directory.rglob("*.tar.gz"))
        for path in tqdm.tqdm(zip_files, desc="Extracting ZIP files"):
            try:
                with zipfile.ZipFile(path, "r") as zip_ref:
                    zip_ref.extractall(path.parent)
                print(f"Extracted {path}")
            except Exception as e:
                print(f"Failed to extract {path}: {e}")
        for path in tqdm.tqdm(tar_files, desc="Extracting TAR files"):
            try:
                with tarfile.open(path, "r:*") as tar_ref:
                    tar_ref.extractall(path.parent)
                print(f"Extracted {path}")
            except Exception as e:
                print(f"Failed to extract {path}: {e}")

    # Use a subdirectory for each dataset
    dataset_subdir = dataset_name.replace("/", "___").replace("-", "_").lower()
    processed_root = root / dataset_subdir

    primary_root: Path = processed_root
    for split in splits:
        if dataset_name == "cifar100":
            if split not in _VALID_SPLITS:
                raise ValueError(f"Unknown split '{split}'. Expected one of {_VALID_SPLITS}.")
            train_flag = split == "train"
            if debug:
                print(f"[DEBUG] Requesting CIFAR100 split='{split}' at {downloads_dir}")
            CIFAR100(root=str(downloads_dir), train=train_flag, download=True)
            primary_root = processed_root
            downloaded[split] = processed_root
        elif dataset_key.startswith("cub200") or dataset_key.startswith("cub"):
            url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
            tar_path = downloads_dir / "CUB_200_2011.tgz"

            if not tar_path.exists():
                print(f"Downloading CUB-200-2011 from {url}...")
                urllib.request.urlretrieve(url, tar_path)
                print("Download complete!")

            extract_dir = downloads_dir / "CUB_200_2011"

            if not extract_dir.exists():
                print("Extracting CUB-200-2011 dataset...")
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(downloads_dir)
                print("Extraction complete!")

            target_cub_dir = processed_root / "CUB_200_2011"
            if not target_cub_dir.exists():
                shutil.copytree(extract_dir, target_cub_dir)

            print("Generating CUB-200 train/val/test splits")
            CUB200Dataset._try_generate_cub200_metadata(processed_root)

            downloaded[split] = processed_root
            primary_root = processed_root
            continue
        else:
            if debug:
                print(
                    f"[DEBUG] Loading HuggingFace dataset '{dataset_name}' split='{split}' cache_dir={downloads_dir}"
                )
            datasets.load_dataset(dataset_name, split=split, cache_dir=str(downloads_dir))
            if snapshot_download:
                target_dir = downloads_dir / f"{dataset_name.replace('/', '___')}"
                target_dir.mkdir(parents=True, exist_ok=True)
                huggingface_hub.snapshot_download(
                    repo_id=dataset_name,
                    repo_type="dataset",
                    revision="main",
                    local_dir=str(target_dir),
                )
                if debug:
                    print(
                        f"[DEBUG] Using snapshot_download for dataset '{dataset_name}' split='{split}'"
                    )
                    print(
                        f"[DEBUG] Downloaded dataset to {target_dir}, extracting archives if any."
                    )
                extract_archives(target_dir)

                split_dirs = ["train", "val", "test"]
                has_split = any((target_dir / s).exists() for s in split_dirs)
                
                if target_dir != processed_root:
                    if processed_root.exists():
                        shutil.rmtree(processed_root)
                    shutil.move(str(target_dir), str(processed_root))
                
                downloaded[split] = processed_root
                primary_root = processed_root
            else:
                downloaded[split] = processed_root
                primary_root = processed_root
    try:
        ensure_standard_metadata(primary_root, dataset_name)
    except FileNotFoundError as exc:
        if debug:
            print(f"[DEBUG] Skipping metadata generation: {exc}")
    return downloaded

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the datasets so wejepa pretraining can reuse it "
            "without hitting the network."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path.cwd() / "data",
        help="Folder that should contain the dataset downloads",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="cifar100",
        help="Name of the dataset to download "
        "(default: cifar100, class dataset: tsbpp/fall2025_deeplearning)",
    )
    parser.add_argument(
        "--snapshot-download",
        action="store_true",
        help="Use snapshot_download from huggingface datasets.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "test"),
        # choices=sorted(_VALID_SPLITS),
        help="Which dataset splits to materialize.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose download logging",
    )
    return parser.parse_args()


def main() -> None:

    args = _parse_args()
    downloads = download(
        args.dataset_root, 
        args.dataset_name, 
        snapshot_download=args.snapshot_download,
        splits=args.splits, 
        debug=args.debug
    )
    for split, root in downloads.items():
        print(f"Split '{split}' available under {root}")


if __name__ == "__main__":
    main()
