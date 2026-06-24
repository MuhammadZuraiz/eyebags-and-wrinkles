#!/usr/bin/env python3
"""
Batch under-eye ROI cropper for DermaLens.

What this script does:
  Processes a folder full of face images and generates two 256×160 under-eye
  crops for every image that passes the quality gate. The crops are saved with
  filenames that encode which eye they came from so you can load them in Label
  Studio for annotation.

Run this on Day 2 BEFORE opening Label Studio.
The output crops are what annotators will grade (0–4 severity).

Usage:
    # Basic — process all .jpg/.png images in a folder
    python scripts/batch_crop.py --input data/raw_faces --output data/crops

    # With spot-check overlays (saves full-face images with ROI boxes drawn on)
    python scripts/batch_crop.py --input data/raw_faces --output data/crops --save-overlays

    # Limit to first 50 images (useful for quickly checking the pipeline works)
    python scripts/batch_crop.py --input data/raw_faces --output data/crops --max 50

Output structure:
    data/crops/
        left/               ← left under-eye crops (256×160)
            img001_left.jpg
            img002_left.jpg
        right/              ← right under-eye crops (256×160)
            img001_right.jpg
        overlays/           ← full-face images with ROI boxes drawn (--save-overlays only)
            img001_overlay.jpg
        crop_log.csv        ← one row per image with success/failure details
        crop_summary.txt    ← overall statistics

Annotation workflow after running this:
  1. Open Label Studio.
  2. Import the left/ folder.
  3. Create a labelling task with choices: severity_grade (0,1,2,3,4) and dark_circles (yes/no).
  4. Annotate. Export as CSV or JSON.
  5. Do the same for right/.
  OR: annotate left and right crops in one task by importing both side-by-side.
  See docs/annotation_guide.md for the recommended Label Studio setup.
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

# Add project root to path so imports work when running from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.face_landmarks import FaceLandmarker
from src.preprocessing.roi_cropper    import RoiCropper, draw_roi_overlay
from src.preprocessing.quality_gate   import QualityGate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ──────────────────────────────────────────────────────────────────────────────
# Core processing
# ──────────────────────────────────────────────────────────────────────────────

def process_image(
    image_path:  Path,
    landmarker:  FaceLandmarker,
    cropper:     RoiCropper,
    quality_gate: QualityGate,
    output_dir:  Path,
    save_overlay: bool,
) -> dict:
    """
    Process a single face image: quality check → landmark detection → crop → save.

    Returns:
        A dict row suitable for writing to crop_log.csv.
    """
    stem = image_path.stem
    row = {
        "image":          image_path.name,
        "quality_ok":     False,
        "landmarks_ok":   False,
        "left_ok":        False,
        "right_ok":       False,
        "left_path":      "",
        "right_path":     "",
        "quality_reasons": "",
        "left_error":     "",
        "right_error":    "",
        "blur_score":     "",
        "pose_yaw":       "",
        "pose_pitch":     "",
        "pose_roll":      "",
    }

    # ── Load image ────────────────────────────────────────────────────────
    img = cv2.imread(str(image_path))
    if img is None:
        row["quality_reasons"] = "imread_failed"
        return row

    # ── Quality gate ──────────────────────────────────────────────────────
    quality = quality_gate.check(img)
    row["blur_score"] = quality.details.get("blur_laplacian", "")
    row["pose_yaw"]   = quality.details.get("pose_yaw_deg", "")
    row["pose_pitch"] = quality.details.get("pose_pitch_deg", "")
    row["pose_roll"]  = quality.details.get("pose_roll_deg", "")

    if not quality.accepted:
        row["quality_reasons"] = "|".join(quality.reasons)
        return row

    row["quality_ok"] = True

    # ── Landmark detection ────────────────────────────────────────────────
    landmarks = landmarker.detect(img)
    if not landmarks.success:
        row["quality_reasons"] = f"landmarks_failed: {landmarks.error_msg}"
        return row

    row["landmarks_ok"] = True

    # ── Crop ROIs ─────────────────────────────────────────────────────────
    rois = cropper.crop(img, landmarks)

    # Save left crop
    if rois.left.success:
        left_path = output_dir / "left" / f"{stem}_left.jpg"
        cv2.imwrite(str(left_path), rois.left.crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        row["left_ok"]   = True
        row["left_path"] = str(left_path.relative_to(output_dir))
    else:
        row["left_error"] = rois.left.error_msg

    # Save right crop
    if rois.right.success:
        right_path = output_dir / "right" / f"{stem}_right.jpg"
        cv2.imwrite(str(right_path), rois.right.crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        row["right_ok"]   = True
        row["right_path"] = str(right_path.relative_to(output_dir))
    else:
        row["right_error"] = rois.right.error_msg

    # ── Overlay (optional) ────────────────────────────────────────────────
    if save_overlay:
        overlay     = draw_roi_overlay(img, rois)
        overlay_path = output_dir / "overlays" / f"{stem}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])

    return row


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch under-eye ROI cropper for DermaLens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",        required=True, help="Folder of face images")
    parser.add_argument("--output",       required=True, help="Output folder for crops")
    parser.add_argument("--save-overlays", action="store_true",
                        help="Save full-face images with ROI boxes drawn on them")
    parser.add_argument("--max",          type=int, default=0,
                        help="Process only the first N images (0 = all)")
    parser.add_argument("--skip-quality", action="store_true",
                        help="Skip the quality gate (not recommended for real data)")
    args = parser.parse_args()

    input_dir  = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()

    if not input_dir.exists():
        print(f"❌ Input folder not found: {input_dir}")
        sys.exit(1)

    # ── Collect image paths ───────────────────────────────────────────────
    image_paths: List[Path] = sorted(
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"❌ No images found in {input_dir}")
        sys.exit(1)

    if args.max > 0:
        image_paths = image_paths[:args.max]

    logger.info(f"Found {len(image_paths)} images in {input_dir}")

    # ── Create output directories ─────────────────────────────────────────
    for sub in ["left", "right", "overlays"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    # ── Initialise pipeline components ───────────────────────────────────
    logger.info("Loading MediaPipe FaceMesh model...")
    landmarker    = FaceLandmarker()
    cropper       = RoiCropper()
    quality_gate  = QualityGate(landmarker=landmarker)

    # ── Process all images ────────────────────────────────────────────────
    log_rows: List[dict] = []
    counts = {
        "total": len(image_paths),
        "quality_fail": 0,
        "landmarks_fail": 0,
        "both_crops_ok": 0,
        "partial_crops": 0,
    }

    t_start = time.time()
    for i, img_path in enumerate(image_paths):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate    = (i + 1) / max(elapsed, 0.001)
            eta     = (len(image_paths) - i - 1) / max(rate, 0.001)
            logger.info(
                f"  [{i+1:>5}/{len(image_paths)}]  "
                f"{rate:.1f} img/s  ETA {eta/60:.1f} min"
            )

        row = process_image(
            image_path   = img_path,
            landmarker   = landmarker,
            cropper      = cropper,
            quality_gate = quality_gate,
            output_dir   = output_dir,
            save_overlay = args.save_overlays,
        )
        log_rows.append(row)

        if not row["quality_ok"]:
            counts["quality_fail"] += 1
        elif not row["landmarks_ok"]:
            counts["landmarks_fail"] += 1
        elif row["left_ok"] and row["right_ok"]:
            counts["both_crops_ok"] += 1
        else:
            counts["partial_crops"] += 1

    elapsed = time.time() - t_start
    logger.info(f"Finished in {elapsed:.1f}s  ({elapsed/len(image_paths)*1000:.0f} ms/image)")

    # ── Write crop_log.csv ────────────────────────────────────────────────
    log_path = output_dir / "crop_log.csv"
    if log_rows:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
        logger.info(f"Log written → {log_path}")

    # ── Write summary ─────────────────────────────────────────────────────
    summary_lines = [
        "DermaLens Batch Crop Summary",
        "=" * 40,
        f"Input folder   : {input_dir}",
        f"Output folder  : {output_dir}",
        f"Total images   : {counts['total']}",
        f"Quality failed : {counts['quality_fail']}  ({counts['quality_fail']/counts['total']*100:.1f}%)",
        f"Landmark fail  : {counts['landmarks_fail']}",
        f"Both crops OK  : {counts['both_crops_ok']}  ← these go into Label Studio",
        f"Partial crops  : {counts['partial_crops']}  ← one eye succeeded",
        f"",
        f"Usable for annotation: {counts['both_crops_ok']} image pairs",
        f"",
        "Next steps:",
        "  1. Open Label Studio",
        "  2. Import data/crops/left/  and  data/crops/right/",
        "  3. Follow docs/annotation_guide.md to set up the labelling interface",
        "  4. Annotate severity (0-4) for each eye + dark circles (yes/no)",
        "  5. Export as CSV and save to data/annotations/",
    ]
    summary_text = "\n".join(summary_lines)
    summary_path = output_dir / "crop_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    # Print summary to console
    print(f"\n{'='*50}")
    print(summary_text)
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
