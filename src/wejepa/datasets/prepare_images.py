"""
prepare_images.py: Utility functions to reorganize datasets into ImageFolder format.

- Can be used as a script or imported as a library.
- Supports both class map and directory-based class inference.
"""

import os
import shutil
import tqdm
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, Set

def find_images(root: Path, exts: Optional[Set[str]] = None):
    """Recursively find all image files under root."""
    if exts is None:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for tqdmdirpath, _, filenames in os.walk(root):
        for fname in tqdm.tqdm(filenames):
            if any(fname.lower().endswith(ext) for ext in exts):
                yield Path(tqdmdirpath) / fname

def prepare_imagefolder(
    input_dir: str,
    output_dir: str,
    class_map: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
    exts: Optional[Set[str]] = None,
) -> Dict[str, int]:
    """
    Reorganize images into ImageFolder format.
    - input_dir: root of raw dataset
    - output_dir: where to write ImageFolder structure
    - class_map: dict mapping image relative paths to class names (optional)
    - dry_run: if True, only print actions
    - exts: set of allowed image extensions
    Returns: dict of class_name -> image count
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_counts = defaultdict(int)

    if class_map:
        for img_rel, class_name in class_map.items():
            src = input_dir / img_rel
            dst = output_dir / class_name / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dry_run:
                print(f"[DRY RUN] {src} -> {dst}")
            else:
                shutil.copy2(src, dst)
            class_counts[class_name] += 1
        return dict(class_counts)

    # Infer class from parent directory
    for img_path in find_images(input_dir, exts=exts):
        class_name = img_path.parent.name
        dst = output_dir / class_name / img_path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dry_run:
            print(f"[DRY RUN] {img_path} -> {dst}")
        else:
            shutil.copy2(img_path, dst)
        class_counts[class_name] += 1
    return dict(class_counts)

def load_class_map(class_map_path: str) -> Dict[str, str]:
    """Load a class map file: <img_rel_path> <class_name> per line."""
    mapping = {}
    with open(class_map_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            mapping[parts[0]] = parts[1]
    return mapping

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Flatten and reorganize dataset into ImageFolder format."
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Path to the raw dataset root (e.g., CUB-200 or similar)."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Path to the output directory in ImageFolder format."
    )
    parser.add_argument(
        "--class-map", type=str, default=None,
        help="Optional: Path to a file mapping image paths to class names (one per line: <img_path> <class>)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print actions without copying files."
    )
    args = parser.parse_args()

    class_map = load_class_map(args.class_map) if args.class_map else None
    counts = prepare_imagefolder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        class_map=class_map,
        dry_run=args.dry_run,
    )
    print("Summary:")
    for cls, count in counts.items():
        print(f"  {cls}: {count} images")
    print("Done.")
