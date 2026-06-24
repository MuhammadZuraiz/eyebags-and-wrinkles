#!/usr/bin/env python3
"""
Download the Face Research Lab London Set (figshare article 5047666).

WHY THIS DATASET:
  - CC BY 4.0 (commercial-compatible attribution license)
  - 102 adults with documented research consent
  - Neutral + smiling frontal photos at 1350x1350 under studio lighting
  - Demographics CSV (age, gender, ethnicity) for fairness slicing

It is the cleanest free seed source for the DermaLens prototype. The script
downloads only the frontal sets (the 3/4 and profile views fail the pose
gate anyway), extracts them, and writes a provenance manifest that
scripts/generate_ls_tasks.py and scripts/prepare_training_csv.py consume.

Usage:
    python scripts/fetch_london_set.py                     # downloads ~90 MB
    python scripts/fetch_london_set.py --out data/raw/london_faces

Output:
    data/raw/london_faces/
        downloads/            raw zips + csv (kept for re-runs)
        images/               all frontal jpgs, flattened
        manifest.csv          one row per image with subject/license/consent

Attribution (CC BY 4.0):
    DeBruine, Lisa; Jones, Benedict (2017). Face Research Lab London Set.
    figshare. https://doi.org/10.6084/m9.figshare.5047666
"""

import argparse
import csv
import json
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

API_URL = "https://api.figshare.com/v2/articles/5047666"

# Only the frontal sets are useful: profiles/3-quarter views fail the yaw gate.
WANTED_FILES = {
    "london_faces_info.csv": "info",
    "neutral_front.zip":     "neutral",
    "smiling_front.zip":     "smiling",
}

LICENSE_STATUS = "cc_by_4_0"
CONSENT_STATUS = "documented_research_consent"
SOURCE_DATASET = "london_faces"


def age_band(age) -> str:
    try:
        a = int(float(age))
    except (TypeError, ValueError):
        return ""
    lo = (a // 10) * 10
    return f"{lo}_{lo + 9}"


def download(url: str, dest: Path, expected_size: int) -> None:
    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"  already downloaded: {dest.name}")
        return
    print(f"  downloading {dest.name} ({expected_size / 1e6:.1f} MB) ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    if expected_size and tmp.stat().st_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {dest.name}: got {tmp.stat().st_size}, "
            f"expected {expected_size}. Delete {tmp} and retry."
        )
    tmp.replace(dest)


def main():
    parser = argparse.ArgumentParser(description="Fetch the London Set from figshare")
    parser.add_argument("--out", default="data/raw/london_faces",
                        help="Output directory (default: data/raw/london_faces)")
    args = parser.parse_args()

    out_dir       = Path(args.out)
    downloads_dir = out_dir / "downloads"
    images_dir    = out_dir / "images"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # ── Query figshare for current file ids/urls ──────────────────────────
    print(f"Querying {API_URL}")
    with urllib.request.urlopen(API_URL) as resp:
        article = json.load(resp)

    license_name = (article.get("license") or {}).get("name", "?")
    print(f"figshare license: {license_name}")
    if "CC BY" not in license_name:
        print("WARNING: license does not look like CC BY — review before using!")

    files = {f["name"]: f for f in article.get("files", [])}
    missing = [n for n in WANTED_FILES if n not in files]
    if missing:
        print(f"ERROR: expected files not in the figshare article: {missing}")
        print(f"Available: {sorted(files)}")
        sys.exit(1)

    # ── Download ──────────────────────────────────────────────────────────
    for name in WANTED_FILES:
        meta = files[name]
        download(meta["download_url"], downloads_dir / name, meta["size"])

    # ── Extract frontal zips, flattened, tagged by expression ─────────────
    expression_of_image = {}
    for name, tag in WANTED_FILES.items():
        if tag == "info":
            continue
        with zipfile.ZipFile(downloads_dir / name) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                fname = Path(member.filename).name
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                # Skip macOS AppleDouble/resource-fork entries (._foo.jpg)
                if fname.startswith("._") or "__MACOSX" in member.filename:
                    continue
                dest = images_dir / fname
                if not dest.exists():
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                expression_of_image[fname] = tag
    print(f"Extracted {len(expression_of_image)} frontal images -> {images_dir}")

    # ── Demographics ──────────────────────────────────────────────────────
    info = {}
    with open(downloads_dir / "london_faces_info.csv", newline="",
              encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            info[row["face_id"].strip().zfill(3)] = row

    # ── Manifest: one row per image, subject id from the filename digits ──
    manifest_path = out_dir / "manifest.csv"
    rows, unmatched = [], []
    for fname in sorted(expression_of_image):
        m = re.match(r"(\d+)", fname)
        if not m:
            unmatched.append(fname)
            continue
        face_id = m.group(1).zfill(3)
        demo = info.get(face_id, {})
        if not demo:
            unmatched.append(fname)
        rows.append({
            "image_file":      fname,
            "subject_id":      f"london_{face_id}",
            "expression":      expression_of_image[fname],
            "age":             demo.get("face_age", ""),
            "age_band":        age_band(demo.get("face_age")),
            "gender":          demo.get("face_gender", ""),
            "ethnicity":       demo.get("face_eth", ""),
            "source_dataset":  SOURCE_DATASET,
            "source_image_id": Path(fname).stem,
            "license_status":  LICENSE_STATUS,
            "consent_status":  CONSENT_STATUS,
        })

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ───────────────────────────────────────────────────────────
    subjects = {r["subject_id"] for r in rows}
    bands = {}
    for r in rows:
        bands[r["age_band"] or "unknown"] = bands.get(r["age_band"] or "unknown", 0) + 1
    print(f"\nManifest -> {manifest_path}")
    print(f"  images:   {len(rows)}")
    print(f"  subjects: {len(subjects)}")
    print(f"  age bands (images): {dict(sorted(bands.items()))}")
    if unmatched:
        print(f"  WARNING: {len(unmatched)} images without demographics match: "
              f"{unmatched[:5]}")
    print("\nNext step:")
    print(f"  python scripts/batch_crop.py --input {images_dir} "
          f"--output data/crops/london --save-overlays")


if __name__ == "__main__":
    main()
