#!/usr/bin/env python3
"""
Build a license-filtered, age-skewed FFHQ subset for DermaLens prototyping.

WHY: the London Set alone (~400 crops) is too small for severity grading, and
its subjects skew young. Eye bags are age-correlated — grade 3-4 examples are
the scarcest and most valuable. FFHQ has 70k in-the-wild 1024x1024 faces with
PER-IMAGE Flickr licenses in its metadata; this script keeps only the clean
buckets (Public Domain / CC0 / U.S. Government Works) and samples skewed old.

LICENSE POSTURE: the FFHQ *package* is CC BY-NC-SA (NVIDIA's curation), even
where individual photos are public domain. Rows from this subset are stamped
license_status=ffhq_pd_prototype_only and MUST NOT ship in commercial weights.

INPUTS YOU DOWNLOAD ONCE (script prints instructions if missing):
  1. ffhq-dataset-v2.json (~255 MB) — per-image metadata with licenses.
     From the official FFHQ Drive folder: https://github.com/NVlabs/ffhq-dataset
  2. (recommended) DCGM ffhq-features-dataset — per-image age estimates:
     git clone https://github.com/DCGM/ffhq-features-dataset
     (pass --features-dir ffhq-features-dataset/json)
  3. (optional) An images1024x1024 mirror folder for --images-dir.
     Without it, selected images are downloaded one by one from the Hugging
     Face mirror (no quota, resumable), falling back to the per-image Google
     Drive URLs (quota-prone) if the mirror is missing a file.

Usage:
    # Primary mode: copy selected ids from a local/Kaggle FFHQ mirror
    python scripts/build_ffhq_subset.py --metadata ffhq-dataset-v2.json ^
        --features-dir ffhq-features-dataset/json ^
        --images-dir D:/datasets/ffhq/images1024x1024 --target 700

    # Top-up mode for grade 3-4 scarcity (only old faces)
    python scripts/build_ffhq_subset.py --metadata ffhq-dataset-v2.json ^
        --features-dir ffhq-features-dataset/json ^
        --images-dir D:/datasets/ffhq/images1024x1024 --target 300 --min-age 55

Output:
    data/raw/ffhq_subset/
        images/          selected 1024x1024 images
        manifest.csv     provenance + age + attribution per image
"""

import argparse
import csv
import json
import random
import shutil
import sys
import time
import urllib.request
from pathlib import Path

SOURCE_DATASET = "ffhq_pd_subset"
LICENSE_STATUS = "ffhq_pd_prototype_only"
CONSENT_STATUS = "none_public_photo"

# Per-image Flickr license buckets considered clean enough for prototyping.
# Exact strings in ffhq-dataset-v2.json: "Public Domain Mark",
# "Public Domain Dedication (CC0)", "United States Government Work".
CLEAN_LICENSE_KEYWORDS = ("public domain", "cc0", "government work")

# Age-skewed sampling: eye bags correlate with age; high grades are scarce.
AGE_QUOTAS = {          # fraction of target
    "45_plus":  0.60,
    "30_44":    0.25,
    "under_30": 0.15,
}


def age_bucket(age) -> str:
    if age is None:
        return "unknown"
    if age >= 45:
        return "45_plus"
    if age >= 30:
        return "30_44"
    return "under_30"


def age_band(age) -> str:
    if age is None:
        return ""
    lo = (int(age) // 10) * 10
    return f"{lo}_{lo + 9}"


def load_metadata(path: Path) -> dict:
    print(f"Loading {path} (this is a ~255 MB JSON; takes a minute) ...")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_ages(features_path: Path, photo_ids) -> dict:
    """DCGM ffhq-features-dataset: one json per image with faceAttributes.age.

    Accepts either the extracted json/ directory or the repo zip itself
    (reading from the zip avoids extracting 70k small files on Windows).
    """
    def parse(raw):
        data = json.loads(raw)
        return float(data[0]["faceAttributes"]["age"]) if data else None

    ages = {}
    if features_path.suffix.lower() == ".zip":
        import zipfile
        with zipfile.ZipFile(features_path) as zf:
            by_name = {Path(n).name: n for n in zf.namelist()
                       if n.endswith(".json") and "/json/" in n}
            for pid in photo_ids:
                member = by_name.get(f"{pid:05d}.json")
                if not member:
                    continue
                try:
                    age = parse(zf.read(member))
                    if age is not None:
                        ages[pid] = age
                except (KeyError, IndexError, ValueError, json.JSONDecodeError):
                    continue
        return ages

    for pid in photo_ids:
        fp = features_path / f"{pid:05d}.json"
        if not fp.exists():
            continue
        try:
            age = parse(fp.read_text(encoding="utf-8"))
            if age is not None:
                ages[pid] = age
        except (KeyError, IndexError, ValueError, json.JSONDecodeError):
            continue
    return ages


def find_image_in_mirror(images_dir: Path, pid: int):
    """FFHQ mirrors store either flat files or 1k-sized subfolders."""
    name = f"{pid:05d}.png"
    candidates = [
        images_dir / name,
        images_dir / f"{(pid // 1000) * 1000:05d}" / name,
        images_dir / name.replace(".png", ".jpg"),
        images_dir / f"{(pid // 1000) * 1000:05d}" / name.replace(".png", ".jpg"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# HF mirror of images1024x1024: Part1..Part7 of 10k PNGs each, no auth/quota.
HF_MIRROR_URL = ("https://huggingface.co/datasets/marcosv/ffhq-dataset"
                 "/resolve/main/Part{part}/{pid:05d}.png")


def download_from_hf(pid: int, dest: Path, retries: int = 3) -> bool:
    url = HF_MIRROR_URL.format(part=pid // 10000 + 1, pid=pid)
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            tmp = dest.with_suffix(".part")
            with urllib.request.urlopen(req, timeout=120) as resp, \
                    open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out)
            if tmp.stat().st_size > 10_000:
                tmp.replace(dest)
                return True
            tmp.unlink(missing_ok=True)
        except Exception as exc:
            print(f"    HF attempt {attempt} failed for {pid:05d}: {exc}")
        time.sleep(2 * attempt)
    return False


def download_from_drive(file_url: str, dest: Path, retries: int = 3) -> bool:
    """Prefer gdown (handles Drive confirm tokens); fall back to urllib."""
    for attempt in range(1, retries + 1):
        try:
            try:
                import gdown
                gdown.download(file_url, str(dest), quiet=True)
            except ImportError:
                req = urllib.request.Request(
                    file_url, headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req, timeout=60) as resp, \
                        open(dest, "wb") as out:
                    shutil.copyfileobj(resp, out)
            if dest.exists() and dest.stat().st_size > 10_000:
                return True                     # Drive quota pages are tiny HTML
            dest.unlink(missing_ok=True)
        except Exception as exc:
            print(f"    attempt {attempt} failed: {exc}")
            dest.unlink(missing_ok=True)
        time.sleep(2 * attempt)
    return False


def main():
    parser = argparse.ArgumentParser(description="Build license-filtered FFHQ subset")
    parser.add_argument("--metadata", required=True,
                        help="Path to ffhq-dataset-v2.json")
    parser.add_argument("--features-dir", default="",
                        help="Path to DCGM ffhq-features-dataset/json directory "
                             "OR the repo zip (age estimates)")
    parser.add_argument("--images-dir", default="",
                        help="Local FFHQ images1024x1024 mirror (primary mode). "
                             "If omitted, downloads selected files from Drive URLs.")
    parser.add_argument("--target", type=int, default=700,
                        help="Number of images to select (default 700)")
    parser.add_argument("--min-age", type=int, default=0,
                        help="Only select faces at or above this age (top-up mode)")
    parser.add_argument("--include-cc-by", action="store_true",
                        help="Also accept CC BY 2.0 images (adds attribution burden)")
    parser.add_argument("--out", default="data/raw/ffhq_subset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta_path = Path(args.metadata)
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found.\n")
        print("Download ffhq-dataset-v2.json from the official FFHQ Drive folder")
        print("linked at https://github.com/NVlabs/ffhq-dataset (~255 MB), then")
        print("re-run with --metadata <path-to-json>.")
        sys.exit(1)

    out_dir    = Path(args.out)
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(meta_path)

    # ── License filter ────────────────────────────────────────────────────
    keywords = CLEAN_LICENSE_KEYWORDS + (("cc by 2.0",) if args.include_cc_by else ())
    candidates = {}
    for key, item in metadata.items():
        info = item.get("metadata", {})
        lic  = (info.get("license") or "").lower()
        if any(k in lic for k in keywords):
            candidates[int(key)] = info
    print(f"License filter: {len(candidates)} / {len(metadata)} images pass "
          f"({'incl. CC BY 2.0' if args.include_cc_by else 'PD/CC0/USGov only'})")
    if not candidates:
        print("ERROR: no images passed the license filter — check the metadata file.")
        sys.exit(1)

    # ── Age estimates ─────────────────────────────────────────────────────
    ages = {}
    if args.features_dir:
        fdir = Path(args.features_dir)
        if fdir.exists():
            print("Loading age estimates ...")
            ages = load_ages(fdir, candidates.keys())
            print(f"  ages available for {len(ages)} / {len(candidates)} candidates")
        else:
            print(f"WARNING: --features-dir {fdir} not found; sampling without ages.")
    else:
        print("WARNING: no --features-dir given. Age-skewed sampling is the main")
        print("lever against grade 3-4 scarcity — strongly consider providing it.")

    # ── Sampling ──────────────────────────────────────────────────────────
    rng = random.Random(args.seed)
    if args.min_age > 0:
        pool = [pid for pid in candidates if ages.get(pid, -1) >= args.min_age]
        rng.shuffle(pool)
        selected = pool[: args.target]
        print(f"Top-up mode (age >= {args.min_age}): selected {len(selected)}")
    elif ages:
        by_bucket = {b: [] for b in AGE_QUOTAS}
        unknown = []
        for pid in candidates:
            a = ages.get(pid)
            (by_bucket[age_bucket(a)] if a is not None else unknown).append(pid)
        for bucket in by_bucket.values():
            rng.shuffle(bucket)
        rng.shuffle(unknown)

        selected, shortfall = [], 0
        for bucket_name, frac in AGE_QUOTAS.items():
            want = int(round(args.target * frac))
            got  = by_bucket[bucket_name][:want]
            shortfall += max(0, want - len(got))
            selected += got
            print(f"  {bucket_name}: wanted {want}, got {len(got)} "
                  f"(pool {len(by_bucket[bucket_name])})")
        # Backfill shortfall from unknown ages, then youngest-first leftovers
        backfill = unknown + [
            pid for b in ("under_30", "30_44", "45_plus")
            for pid in by_bucket[b][int(round(args.target * AGE_QUOTAS[b])):]
        ]
        selected += backfill[:shortfall]
    else:
        pool = list(candidates)
        rng.shuffle(pool)
        selected = pool[: args.target]

    print(f"Selected {len(selected)} images")

    # ── Materialize images ────────────────────────────────────────────────
    mirror = Path(args.images_dir) if args.images_dir else None
    copied, failed = 0, []
    for i, pid in enumerate(selected):
        dest = images_out / f"ffhq_{pid:05d}.png"
        if dest.exists() or dest.with_suffix(".jpg").exists():
            copied += 1
            continue
        if mirror:
            src = find_image_in_mirror(mirror, pid)
            if src is None:
                failed.append(pid)
                continue
            shutil.copy2(src, dest.with_suffix(src.suffix))
            copied += 1
        else:
            if (i + 1) % 25 == 0 or i == 0:
                print(f"  [{i + 1}/{len(selected)}] downloading {pid:05d} ...",
                      flush=True)
            if download_from_hf(pid, dest):
                copied += 1
                continue
            # Fall back to the official per-image Drive URL (quota-prone)
            file_url = metadata[str(pid)].get("image", {}).get("file_url")
            if file_url and download_from_drive(file_url, dest):
                copied += 1
            else:
                failed.append(pid)
    print(f"Materialized {copied} images ({len(failed)} unavailable)")
    if failed and not mirror:
        print("Some files failed from both the HF mirror and Drive — the script")
        print("is resumable, just re-run it later to pick up the stragglers.")

    # ── Manifest ──────────────────────────────────────────────────────────
    available = {p.stem: p.name for p in images_out.iterdir()
                 if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
    manifest_path = out_dir / "manifest.csv"
    rows = []
    for pid in sorted(selected):
        stem = f"ffhq_{pid:05d}"
        if stem not in available:
            continue
        info = candidates[pid]
        a = ages.get(pid)
        rows.append({
            "image_file":      available[stem],
            "subject_id":      stem,   # FFHQ has no identity metadata (see docs)
            "age_est":         f"{a:.0f}" if a is not None else "",
            "age_band":        age_band(a),
            "license":         info.get("license", ""),
            "license_status":  LICENSE_STATUS,
            "consent_status":  CONSENT_STATUS,
            "source_dataset":  SOURCE_DATASET,
            "source_image_id": stem,
            "photo_url":       info.get("photo_url", ""),
            "author":          info.get("author", ""),
        })
    # Merge with any existing manifest so top-up runs (--min-age) extend it
    # instead of orphaning previously selected images. New rows win.
    merged = {}
    if manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for old in csv.DictReader(f):
                merged[old["image_file"]] = old
    for r in rows:
        merged[r["image_file"]] = r
    rows = [merged[k] for k in sorted(merged)]

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    bands = {}
    for r in rows:
        bands[r["age_band"] or "unknown"] = bands.get(r["age_band"] or "unknown", 0) + 1
    print(f"\nManifest -> {manifest_path}  ({len(rows)} rows)")
    print(f"  age bands: {dict(sorted(bands.items()))}")
    print("\nNext step:")
    print(f"  python scripts/batch_crop.py --input {images_out} "
          f"--output data/crops/ffhq --save-overlays")


if __name__ == "__main__":
    main()
