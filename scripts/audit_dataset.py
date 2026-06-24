#!/usr/bin/env python3
"""
DermaLens Dataset Audit Script
================================
Run this against each of your existing YOLO datasets BEFORE writing any model code.
It tells you exactly what you have, what's missing, and what might cause problems.

Usage:
    python scripts/audit_dataset.py --dataset /path/to/dark_circles --name dark_circles
    python scripts/audit_dataset.py --dataset /path/to/facial_skin  --name facial_skin
    python scripts/audit_dataset.py --dataset /path/to/acne_v5      --name acne_v5

What it checks:
    - Total image count per split (train/valid/test)
    - Label file count and empty label files
    - Class distribution
    - Annotation type (bbox vs polygon)
    - Image dimensions
    - Blur scores (image quality)
    - Cross-split duplicates (LEAKAGE CHECK)
    - Within-split duplicates
    - Whether dark_circles and eye_bags labels are mixed together

Output:
    - audit_results/<name>_audit.json   — full structured report
    - Console summary with action items
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import yaml

# --- Optional deps (installed via requirements.txt) ---
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠️  PIL not found. Run: pip install Pillow")

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("⚠️  imagehash not found. Duplicate detection disabled. Run: pip install imagehash")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️  OpenCV not found. Blur detection disabled. Run: pip install opencv-python")


# ─────────────────────────────────────────────
# Label parsing
# ─────────────────────────────────────────────

def parse_yolo_label(label_path: Path) -> List[Dict]:
    """Parse a YOLO label file. Returns list of annotation dicts."""
    if not label_path.exists():
        return []
    content = label_path.read_text().strip()
    if not content:
        return []  # Empty file = background / true negative

    annotations = []
    for line in content.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
            values = [float(v) for v in parts[1:]]
            n = len(values)
            if n == 4:
                ann_type = "bbox"
            elif n >= 6 and n % 2 == 0:
                ann_type = "polygon"
            elif n == 0:
                ann_type = "class_only"
            else:
                ann_type = f"unknown({n}_values)"
            annotations.append({"class_id": class_id, "type": ann_type})
        except (ValueError, IndexError):
            continue
    return annotations


# ─────────────────────────────────────────────
# Image analysis helpers
# ─────────────────────────────────────────────

def compute_blur_score(image_path: Path) -> Optional[float]:
    """Laplacian variance: HIGHER = sharper. Below ~50 is visibly blurry."""
    if not HAS_CV2:
        return None
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception:
        return None


def compute_phash(image_path: Path) -> Optional[str]:
    """Perceptual hash for duplicate detection."""
    if not HAS_IMAGEHASH or not HAS_PIL:
        return None
    try:
        return str(imagehash.phash(Image.open(image_path)))
    except Exception:
        return None


# ─────────────────────────────────────────────
# Core audit logic
# ─────────────────────────────────────────────

def find_image_and_label_dirs(split_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Handle the two common YOLO dataset structures:
      Structure A: split_dir/images/  + split_dir/labels/
      Structure B: split_dir/ contains images directly, labels in sibling labels/split/
    """
    # Structure A: explicit images/ subfolder
    if (split_dir / "images").exists():
        img_dir = split_dir / "images"
        label_dir = split_dir / "labels"
        return img_dir, label_dir if label_dir.exists() else None

    # Structure B: images are directly in split_dir
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    has_images = any(f.suffix.lower() in IMAGE_EXTENSIONS for f in split_dir.iterdir() if f.is_file())
    if has_images:
        img_dir = split_dir
        # Look for labels next to images
        label_dir = split_dir.parent.parent / "labels" / split_dir.name
        if not label_dir.exists():
            label_dir = split_dir  # labels in same folder
        return img_dir, label_dir

    return None, None


def audit_split(split_dir: Path, class_names: List[str]) -> Dict:
    """Audit a single data split (train/valid/test)."""
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

    img_dir, label_dir = find_image_and_label_dirs(split_dir)
    if img_dir is None:
        return {"error": f"No images found in {split_dir}"}

    image_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])

    result = {
        "total_images": len(image_files),
        "total_labels": 0,
        "empty_labels": 0,
        "missing_labels": 0,
        "class_distribution": defaultdict(int),
        "annotation_types": Counter(),
        "image_dimensions": [],
        "blur_scores": [],
        "phashes": {},
        "corrupt_images": [],
    }

    for img_path in image_files:
        # --- Label ---
        if label_dir and label_dir.exists():
            label_path = label_dir / (img_path.stem + ".txt")
        else:
            label_path = img_dir / (img_path.stem + ".txt")

        if label_path.exists():
            result["total_labels"] += 1
            annotations = parse_yolo_label(label_path)
            if not annotations:
                result["empty_labels"] += 1
            else:
                for ann in annotations:
                    cid = ann["class_id"]
                    name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
                    result["class_distribution"][name] += 1
                    result["annotation_types"][ann["type"]] += 1
        else:
            result["missing_labels"] += 1

        # --- Dimensions ---
        if HAS_PIL:
            try:
                with Image.open(img_path) as img:
                    result["image_dimensions"].append(img.size)  # (W, H)
            except Exception:
                result["corrupt_images"].append(img_path.name)

        # --- Blur (sample up to 300 images for speed) ---
        if HAS_CV2 and len(result["blur_scores"]) < 300:
            score = compute_blur_score(img_path)
            if score is not None:
                result["blur_scores"].append(score)

        # --- Perceptual hash (all images) ---
        if HAS_IMAGEHASH:
            ph = compute_phash(img_path)
            if ph:
                result["phashes"][img_path.name] = ph

    return result


def find_duplicates(split_results: Dict) -> Tuple[List[Tuple], Dict]:
    """
    Returns:
        cross_dupes: list of (split1, file1, split2, file2) — LEAKAGE
        within_dupes: {split_name: {hash: [files]}} — redundant data
    """
    if not HAS_IMAGEHASH:
        return [], {}

    # Cross-split
    hash_registry = {}  # hash → (split_name, filename)
    cross_dupes = []
    for split_name, data in split_results.items():
        for fname, ph in data.get("phashes", {}).items():
            if ph in hash_registry:
                cross_dupes.append((hash_registry[ph][0], hash_registry[ph][1], split_name, fname))
            else:
                hash_registry[ph] = (split_name, fname)

    # Within-split
    within_dupes = {}
    for split_name, data in split_results.items():
        hash_to_files = defaultdict(list)
        for fname, ph in data.get("phashes", {}).items():
            hash_to_files[ph].append(fname)
        groups = {ph: fs for ph, fs in hash_to_files.items() if len(fs) > 1}
        if groups:
            within_dupes[split_name] = groups

    return cross_dupes, within_dupes


EYE_BAG_KEYWORDS    = {'eye_bag', 'eyebag', 'under_eye', 'puffy', 'puffiness', 'bags_under'}
DARK_CIRCLE_KEYWORDS = {'dark_circle', 'periorbital', 'dark_under', 'hyperpig'}


def detect_label_mixing(all_classes: Dict) -> Dict:
    """Check if eye bag and dark circle labels are mixed — these must be separate."""
    bags, darks = {}, {}
    for cls, count in all_classes.items():
        lower = cls.lower().replace(' ', '_')
        if any(k in lower for k in EYE_BAG_KEYWORDS):
            bags[cls] = count
        if any(k in lower for k in DARK_CIRCLE_KEYWORDS):
            darks[cls] = count
    return {
        "eye_bag_classes": bags,
        "dark_circle_classes": darks,
        "mixing_suspected": bool(bags and darks),
    }


# ─────────────────────────────────────────────
# Report printing
# ─────────────────────────────────────────────

def print_report(name: str, yaml_info: Dict, split_results: Dict,
                 cross_dupes: List, within_dupes: Dict, output_dir: Path):

    SEP = "=" * 60
    print(f"\n{SEP}")
    print(f"  DATASET AUDIT: {name}")
    print(SEP)

    if yaml_info:
        classes = yaml_info.get('names', [])
        print(f"\n  Classes ({len(classes)}): {', '.join(classes)}")

    all_class_dist = defaultdict(int)
    action_items = []

    for split_name, data in split_results.items():
        if "error" in data:
            print(f"\n  ❌ {split_name.upper()}: {data['error']}")
            continue

        n_img   = data["total_images"]
        n_lbl   = data["total_labels"]
        n_empty = data["empty_labels"]
        n_miss  = data["missing_labels"]

        print(f"\n  {'─'*45}")
        print(f"  Split: {split_name.upper()}")
        print(f"  {'─'*45}")
        print(f"  📸 Images        : {n_img:>6}")
        print(f"  🏷️  Label files  : {n_lbl:>6}")

        miss_icon = "⚠️ " if n_miss > n_img * 0.02 else "✅"
        print(f"  {miss_icon} Missing labels : {n_miss:>6}  {'← No .txt for these images!' if n_miss > 0 else ''}")

        empty_pct = n_empty / max(n_lbl, 1) * 100
        empty_icon = "❓" if empty_pct > 20 else "✅"
        print(f"  {empty_icon} Empty labels  : {n_empty:>6}  ({empty_pct:.0f}% of labeled)  {'← True negatives or unannotated?' if n_empty > 0 else ''}")

        if data["annotation_types"]:
            types_str = "  ".join(f"{t}:{n}" for t, n in sorted(data["annotation_types"].items()))
            print(f"  📐 Ann types     : {types_str}")
            if "bbox" in data["annotation_types"] and "polygon" in data["annotation_types"]:
                action_items.append(f"⚠️  {split_name}: Mixed bbox+polygon — convert all to polygon for segmentation")

        if data["class_distribution"]:
            print(f"\n  Class counts:")
            for cls, cnt in sorted(data["class_distribution"].items()):
                print(f"    {cls:<35} : {cnt:>5}")
                all_class_dist[cls] += cnt

        if data["image_dimensions"]:
            ws = [d[0] for d in data["image_dimensions"]]
            hs = [d[1] for d in data["image_dimensions"]]
            print(f"\n  Image W: {min(ws)}–{max(ws)}px  (median {int(np.median(ws))}px)")
            print(f"  Image H: {min(hs)}–{max(hs)}px  (median {int(np.median(hs))}px)")

        if data["blur_scores"]:
            scores = data["blur_scores"]
            very_blurry = sum(1 for s in scores if s < 50)
            print(f"  Blur (Laplacian var):  median={np.median(scores):.0f}  min={min(scores):.0f}  max={max(scores):.0f}")
            if very_blurry > 0:
                pct = very_blurry / len(scores) * 100
                print(f"  ⚠️  {very_blurry} very blurry images ({pct:.0f}% of sampled)")

        if data["corrupt_images"]:
            print(f"  ❌ Corrupt images: {len(data['corrupt_images'])}")

        # Action items for this split
        if n_miss > n_img * 0.05:
            action_items.append(f"❗ {split_name}: {n_miss} images have no label file — is annotation complete?")
        if n_empty > n_lbl * 0.40:
            action_items.append(f"❓ {split_name}: {n_empty} empty labels ({empty_pct:.0f}%) — confirm these are true negatives, not forgotten annotations")

    # Overall class distribution
    if all_class_dist:
        print(f"\n  {'─'*45}")
        print(f"  OVERALL CLASS DISTRIBUTION (all splits combined)")
        print(f"  {'─'*45}")
        total_anns = sum(all_class_dist.values())
        for cls, cnt in sorted(all_class_dist.items(), key=lambda x: -x[1]):
            bar = "█" * int(cnt / max(total_anns, 1) * 30)
            print(f"  {cls:<35} : {cnt:>5}  {bar}")

    # Label mixing check
    mixing = detect_label_mixing(dict(all_class_dist))
    if mixing["mixing_suspected"]:
        print(f"\n  ⚠️  LABEL MIXING: Eye bag classes {mixing['eye_bag_classes']} and dark circle classes {mixing['dark_circle_classes']} are in the same dataset.")
        print(f"     These are DIFFERENT concerns — do not use them interchangeably as labels.")
        action_items.append("⚠️  Label mixing: separate eye_bag and dark_circle annotations before training")
    elif mixing["eye_bag_classes"]:
        print(f"\n  ✅ Eye bag classes found: {mixing['eye_bag_classes']}")
    elif mixing["dark_circle_classes"]:
        print(f"\n  ℹ️  Dark circle classes (NOT eye bags): {mixing['dark_circle_classes']}")
    else:
        print(f"\n  ℹ️  No eye bag or dark circle class names auto-detected. Review class names manually.")

    # Duplicates
    print(f"\n  {'─'*45}")
    print(f"  DUPLICATE ANALYSIS")
    print(f"  {'─'*45}")
    if not HAS_IMAGEHASH:
        print("  Skipped (install imagehash to enable)")
    else:
        if cross_dupes:
            print(f"  🚨 CROSS-SPLIT DUPLICATES: {len(cross_dupes)}  ← DATA LEAKAGE")
            for s1, f1, s2, f2 in cross_dupes[:8]:
                print(f"     {s1}/{f1}  ↔  {s2}/{f2}")
            if len(cross_dupes) > 8:
                print(f"     ... and {len(cross_dupes)-8} more")
            action_items.append(f"🚨 {len(cross_dupes)} cross-split duplicates (leakage) — re-split by participant ID")
        else:
            print("  ✅ No cross-split duplicates")

        within_total = sum(len(fs)-1 for ds in within_dupes.values() for fs in ds.values())
        if within_total > 0:
            print(f"  ⚠️  Within-split duplicates: {within_total} extra copies")
            for sn, dupes in within_dupes.items():
                print(f"     {sn}: {len(dupes)} duplicate groups")
            action_items.append(f"⚠️  {within_total} duplicate images within splits — deduplicate before training")
        else:
            print("  ✅ No within-split duplicates")

    # Action items
    print(f"\n  {'='*45}")
    print(f"  ACTION ITEMS")
    print(f"  {'='*45}")
    if not action_items:
        print("  ✅ No critical issues found. Dataset looks clean.\n")
    else:
        for item in action_items:
            print(f"  {item}")
    print()

    # ─── Save JSON ───
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{name}_audit.json"

    json_out = {
        "dataset": name,
        "class_names": yaml_info.get("names", []) if yaml_info else [],
        "splits": {},
        "cross_split_duplicates": len(cross_dupes),
        "action_items": action_items,
    }
    for sn, data in split_results.items():
        if "error" in data:
            json_out["splits"][sn] = data
            continue
        json_out["splits"][sn] = {
            "total_images": data["total_images"],
            "total_labels": data["total_labels"],
            "empty_labels": data["empty_labels"],
            "missing_labels": data["missing_labels"],
            "class_distribution": dict(data["class_distribution"]),
            "annotation_types": dict(data["annotation_types"]),
            "blur_median": float(np.median(data["blur_scores"])) if data["blur_scores"] else None,
        }

    with open(report_path, 'w') as f:
        json.dump(json_out, f, indent=2)
    print(f"  📄 Report saved → {report_path}\n")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Audit a YOLO-format dataset for the DermaLens eye-bag model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/audit_dataset.py --dataset ~/datasets/dark_circles    --name dark_circles
  python scripts/audit_dataset.py --dataset ~/datasets/facial_skin     --name facial_skin
  python scripts/audit_dataset.py --dataset ~/datasets/acne_v5         --name acne_v5
        """
    )
    parser.add_argument("--dataset", required=True, help="Path to the dataset root directory")
    parser.add_argument("--name",    required=True, help="Short name for this dataset (used in output filenames)")
    parser.add_argument("--output",  default="./audit_results", help="Directory to save JSON reports (default: ./audit_results)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser()
    output_dir   = Path(args.output)

    if not dataset_path.exists():
        print(f"❌ Path not found: {dataset_path}")
        sys.exit(1)

    print(f"\n🔍 Auditing: {args.name}")
    print(f"   Path: {dataset_path.absolute()}")

    # Load data.yaml
    yaml_info = {}
    for yaml_candidate in [dataset_path / "data.yaml", dataset_path / "dataset.yaml"]:
        if yaml_candidate.exists():
            with open(yaml_candidate) as f:
                yaml_info = yaml.safe_load(f) or {}
            print(f"   ✅ Found: {yaml_candidate.name}")
            break
    else:
        print("   ⚠️  No data.yaml found — class names will be shown as class_0, class_1, ...")

    class_names = yaml_info.get('names', [])

    # Detect splits
    splits_found = {}
    # Check for standard YOLO structure: dataset/train/, dataset/valid/, dataset/test/
    for split in ['train', 'valid', 'val', 'test']:
        candidate = dataset_path / split
        if candidate.exists():
            splits_found[split] = candidate
    # Check nested: dataset/images/train/
    if not splits_found:
        img_root = dataset_path / "images"
        if img_root.exists():
            for split in ['train', 'valid', 'val', 'test']:
                candidate = img_root / split
                if candidate.exists():
                    splits_found[split] = dataset_path / split
    # Fallback: treat whole folder as one dataset
    if not splits_found:
        print("   ℹ️  No train/valid/test subfolders found — treating as single dataset")
        splits_found["all"] = dataset_path

    print(f"   Splits found: {', '.join(splits_found.keys())}")
    print(f"\n   Running audit...\n")

    split_results = {}
    for split_name, split_dir in splits_found.items():
        print(f"   Scanning {split_name}...", end='', flush=True)
        split_results[split_name] = audit_split(split_dir, class_names)
        n = split_results[split_name].get("total_images", "?")
        print(f" {n} images")

    cross_dupes, within_dupes = find_duplicates(split_results)
    print_report(args.name, yaml_info, split_results, cross_dupes, within_dupes, output_dir)


if __name__ == "__main__":
    main()
