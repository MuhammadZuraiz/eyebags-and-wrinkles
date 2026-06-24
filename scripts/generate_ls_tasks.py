#!/usr/bin/env python3
"""
Generate Label Studio tasks (one task per eye crop) from batch_crop output.

Joins crop_log.csv (which crops succeeded) with the source manifest.csv
(subject/license/consent/age provenance) and emits the task JSON format from
docs/annotation_guide.md section 3: each task shows the full-face image for
context plus the 256x160 eye crop to grade, with provenance prefilled so
scripts/prepare_training_csv.py can carry it into the training CSV.

Usage:
    python scripts/generate_ls_tasks.py ^
        --crops data/crops/london ^
        --manifest data/raw/london_faces/manifest.csv ^
        --faces data/raw/london_faces/images ^
        --output data/tasks/london_tasks.json

    # Calibration batch: a reproducible random sample annotated twice
    python scripts/generate_ls_tasks.py ^
        --crops data/crops/london --manifest data/raw/london_faces/manifest.csv ^
        --faces data/raw/london_faces/images ^
        --output data/tasks/calibration_tasks.json --sample 80

Label Studio must serve local files with LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT
pointed at the directory given by --serving-root (default: data). Task URLs are
written relative to it as /data/local-files/?d=<relative path>.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def ls_url(path: Path, serving_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(serving_root.resolve())
    except ValueError:
        raise SystemExit(
            f"ERROR: {path} is not under the serving root {serving_root}.\n"
            "Label Studio can only serve files below "
            "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT. Pass a different "
            "--serving-root or move the data."
        )
    return "/data/local-files/?d=" + rel.as_posix()


def main():
    parser = argparse.ArgumentParser(description="Generate Label Studio eye-crop tasks")
    parser.add_argument("--crops", required=True,
                        help="batch_crop output dir (contains crop_log.csv, left/, right/)")
    parser.add_argument("--manifest", required=True,
                        help="Source manifest.csv (from fetch_london_set.py / build_ffhq_subset.py)")
    parser.add_argument("--faces", required=True,
                        help="Directory with the original full-face images")
    parser.add_argument("--output", required=True, help="Output task JSON path")
    parser.add_argument("--serving-root", default="data",
                        help="Label Studio LOCAL_FILES_DOCUMENT_ROOT (default: data)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Emit only N randomly sampled tasks (calibration batch)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    crops_dir    = Path(args.crops)
    faces_dir    = Path(args.faces)
    serving_root = Path(args.serving_root)
    log_path     = crops_dir / "crop_log.csv"
    if not log_path.exists():
        raise SystemExit(f"ERROR: {log_path} not found — run scripts/batch_crop.py first.")

    crop_log = pd.read_csv(log_path).fillna("")
    manifest = pd.read_csv(args.manifest).fillna("")
    by_image = {row["image_file"]: row for _, row in manifest.iterrows()}

    tasks, skipped = [], 0
    for _, row in crop_log.iterrows():
        demo = by_image.get(row["image"])
        if demo is None:
            skipped += 1
            continue
        face_path = faces_dir / row["image"]
        for eye in ("left", "right"):
            rel = row.get(f"{eye}_path", "")
            if not rel or not row.get(f"{eye}_ok"):
                continue
            crop_path = crops_dir / rel
            if not crop_path.exists():
                skipped += 1
                continue
            tasks.append({
                "data": {
                    "face_image":      ls_url(face_path, serving_root),
                    "eye_crop":        ls_url(crop_path, serving_root),
                    "subject_id":      demo["subject_id"],
                    "source_image_id": demo.get("source_image_id", Path(row["image"]).stem),
                    "eye_side":        eye,
                    "source_dataset":  demo["source_dataset"],
                    "license_status":  demo["license_status"],
                    "consent_status":  demo.get("consent_status", "unspecified"),
                    "age_band":        demo.get("age_band", ""),
                }
            })

    if args.sample > 0:
        rng = random.Random(args.seed)
        rng.shuffle(tasks)
        tasks = tasks[: args.sample]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tasks, indent=2), encoding="utf-8")

    subjects = {t["data"]["subject_id"] for t in tasks}
    print(f"Wrote {len(tasks)} tasks ({len(subjects)} subjects) -> {out}")
    if skipped:
        print(f"  skipped {skipped} crop-log rows without manifest/crop match")
    print("\nIn Label Studio: create the project per docs/annotation_guide.md,")
    print(f"set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={serving_root.resolve()},")
    print("then import this JSON file.")


if __name__ == "__main__":
    main()
