#!/usr/bin/env python3
"""
Package code + data into one zip for Colab training.

Produces dermalens_bundle_<version>.zip containing:
    code:  src/ scripts/ configs/ tests/ requirements.txt
    data:  data/splits/ + the crop folders referenced by the split CSVs
           (+ data/smoke for the sanity cell)

Upload the zip to Google Drive, then follow notebooks/colab_train.ipynb.
Re-run with a bumped --version after any re-annotation or re-split.

Usage:
    python scripts/make_colab_bundle.py --version v1
    python scripts/make_colab_bundle.py --version v2 --splits data/splits
"""

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

CODE_DIRS  = ["src", "scripts", "configs", "tests"]
CODE_FILES = ["requirements.txt"]


def make_writer(zf: zipfile.ZipFile):
    """Returns a write function that skips already-added archive names."""
    seen = set()

    def write(path: Path, arcname: Path):
        key = arcname.as_posix()
        if key in seen:
            return
        seen.add(key)
        zf.write(path, arcname)

    return write


def add_tree(write, root: Path, base: Path):
    for p in sorted(root.rglob("*")):
        if p.is_dir() or "__pycache__" in p.parts:
            continue
        write(p, p.relative_to(base))


def main():
    parser = argparse.ArgumentParser(description="Build the Colab training bundle")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--splits",  default="data/splits",
                        help="Split CSV folder to include (default: data/splits)")
    parser.add_argument("--out",     default="",
                        help="Output zip (default: dermalens_bundle_<version>.zip)")
    args = parser.parse_args()

    repo = Path(__file__).parent.parent
    out  = Path(args.out) if args.out else repo / f"dermalens_bundle_{args.version}.zip"

    splits_dir = repo / args.splits
    split_csvs = sorted(splits_dir.glob("*.csv")) if splits_dir.exists() else []

    # Collect every crop file referenced by the split CSVs (paths are stored
    # relative to the repo root, e.g. data/crops/london/left/001_03_left.jpg)
    crop_files, missing = set(), []
    for csv in split_csvs:
        for p in pd.read_csv(csv)["image_path"]:
            fp = repo / p
            (crop_files.add(fp) if fp.exists() else missing.append(str(p)))

    if missing:
        print(f"ERROR: {len(missing)} crop paths in the split CSVs do not exist, "
              f"e.g. {missing[:3]}")
        print("Fix the splits before bundling — Colab training would fail fast.")
        sys.exit(1)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        write = make_writer(zf)
        for d in CODE_DIRS:
            add_tree(write, repo / d, repo)
        for f in CODE_FILES:
            write(repo / f, Path(f))
        add_tree(write, repo / "data" / "smoke", repo)   # sanity-cell data
        for csv in split_csvs:
            write(csv, csv.relative_to(repo))
        for fp in sorted(crop_files):
            write(fp, fp.relative_to(repo))

    size_mb = out.stat().st_size / 1e6
    print(f"Bundle -> {out}  ({size_mb:.0f} MB)")
    print(f"  split CSVs: {len(split_csvs)}   crops: {len(crop_files)}")
    print("\nUpload to Google Drive, then open notebooks/colab_train.ipynb in Colab.")


if __name__ == "__main__":
    main()
