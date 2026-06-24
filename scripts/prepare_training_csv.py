#!/usr/bin/env python3
"""
Convert Label Studio annotation exports into the flat per-crop training CSV.

WHY THIS EXISTS:
  In Label Studio you annotate each CROP individually (one task = one crop
  image, you pick severity 0-4 + dark circles yes/no). The export is messy
  JSON or a wide CSV. This script flattens it into the exact format that
  EyeBagDataset expects:

      image_path,severity,dark_circles,presence,eye,subject_id,
      source_dataset,source_image_id,license_status,consent_status,
      quality_reject,makeup_suspected,annotation_confidence,annotator_id

  It also derives:
    - eye        from the filename suffix (_left.jpg / _right.jpg from batch_crop.py)
    - subject_id from the filename stem (img001_left.jpg → img001)
                 ⚠️  This assumes ONE IMAGE PER PERSON. If you know the same
                 person appears in multiple files, build a proper mapping file
                 (see --subject-map below) — otherwise your splits will leak.

Supported inputs:
  1. Label Studio CSV export (Export → CSV) with columns containing at least:
       image  (or "image_path" / "data" — auto-detected)
       severity (your Choices field name, auto-detected from common names)
       dark_circles
  2. Label Studio JSON export (Export → JSON) — the standard tasks format.
  3. A hand-built simple CSV that already has image_path + severity columns
     (it will just be normalised and validated).

Usage:
    # From Label Studio CSV export
    python scripts/prepare_training_csv.py \
        --input  exports/labelstudio_export.csv \
        --output data/annotations/all_annotations.csv \
        --source-dataset london_faces \
        --license-status cc_by_4_0 \
        --consent-status documented_research_consent

    # From Label Studio JSON export
    python scripts/prepare_training_csv.py \
        --input  exports/labelstudio_export.json \
        --output data/annotations/all_annotations.csv

    # With a subject mapping (filename_stem → subject_id), for multi-photo subjects
    python scripts/prepare_training_csv.py \
        --input exports/export.csv \
        --output data/annotations/all_annotations.csv \
        --subject-map data/subject_map.csv

    subject_map.csv format:
        filename_stem,subject_id
        img001,subj_A
        img002,subj_A      ← same person, two photos
        img003,subj_B
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# Windows consoles default to cp1252, which crashes on the status emojis below.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# Column-name candidates for auto-detection in Label Studio exports
IMAGE_COL_CANDIDATES    = [
    "eye_crop", "crop_image", "crop_path", "image_path", "image", "img", "data", "$image", "ocr"
]
SEVERITY_COL_CANDIDATES = ["severity", "severity_grade", "grade", "eye_bag_severity", "choice"]
DARK_COL_CANDIDATES     = ["dark_circles", "darkcircles", "dark_circle", "dc"]
REJECT_COL_CANDIDATES   = ["quality_reject", "reject", "unusable", "quality"]
MAKEUP_COL_CANDIDATES   = ["makeup_suspected", "makeup", "makeup_detected_or_suspected"]
CONF_COL_CANDIDATES     = ["annotation_confidence", "confidence", "label_confidence"]
ANNOTATOR_COL_CANDIDATES = ["annotator_id", "annotator", "completed_by", "created_by"]
SUBJECT_COL_CANDIDATES  = ["subject_id", "subject", "participant_id", "person_id"]
SOURCE_DATASET_COL_CANDIDATES = ["source_dataset", "dataset", "source"]
SOURCE_IMAGE_COL_CANDIDATES = ["source_image_id", "source_image", "face_image", "face_image_id"]
LICENSE_COL_CANDIDATES  = ["license_status", "license", "usage_terms"]
CONSENT_COL_CANDIDATES  = ["consent_status", "consent", "release_status"]


def _find_col(df: pd.DataFrame, candidates) -> str:
    """Find the first matching column name (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return ""


def _clean_image_path(raw: str, ls_root: str = "data") -> str:
    """
    Label Studio prefixes local files with URLs like:
        /data/local-files/?d=crops/left/img001_left.jpg
    The `?d=` path is relative to the LS document root, which this project
    sets to <repo>/data. Prepend `ls_root` so the result is repo-root-relative
    (e.g. data/crops/left/img001_left.jpg) — the convention EyeBagDataset and
    the bundled smoke CSVs use, so training resolves the files with crops_dir
    left null. Pass --ls-root "" if your doc root is the repo root itself.
    """
    raw = str(raw)
    m = re.search(r"[?&]d=([^&]+)", raw)
    if m:
        rel = m.group(1).lstrip("/")
        return f"{ls_root}/{rel}" if ls_root else rel
    # Strip http(s)://host/ prefixes
    raw = re.sub(r"^https?://[^/]+/", "", raw)
    return raw


def _severity_to_int(val) -> int:
    """Handle '2', 2, 'Grade 2', 'moderate' etc."""
    if pd.isna(val):
        return -1
    s = str(val).strip().lower()
    # Direct integer
    m = re.search(r"\d", s)
    if m:
        return int(m.group())
    word_map = {
        "not present": 0, "none": 0, "absent": 0,
        "mild": 1, "moderate": 2, "pronounced": 3,
        "very pronounced": 4, "severe": 4,
    }
    return word_map.get(s, -1)


def _binary(val) -> int:
    s = str(val).strip().lower()
    return 1 if s in {
        "1", "yes", "true", "y", "present", "visible", "suspected",
        "reject", "quality_reject", "unusable",
        "reject - too blurry/dark/occluded to judge",
    } else 0


def _series_or_default(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    """Return a string series from df[col], or a same-length default series."""
    if col:
        return df[col].fillna("").astype(str)
    return pd.Series([default] * len(df), index=df.index, dtype="object")


def _eye_from_filename(path: str) -> str:
    stem = Path(path).stem.lower()
    if stem.endswith("_left"):
        return "left"
    if stem.endswith("_right"):
        return "right"
    return ""


def _subject_from_filename(path: str) -> str:
    """img001_left.jpg → img001  (strip the _left/_right suffix)"""
    stem = Path(path).stem
    return re.sub(r"_(left|right)$", "", stem, flags=re.IGNORECASE)


def load_labelstudio_json(path: Path) -> pd.DataFrame:
    """Parse the standard Label Studio JSON export into a flat DataFrame."""
    tasks = json.loads(path.read_text())
    rows = []
    for task in tasks:
        image = ""
        data = task.get("data", {})
        for key in IMAGE_COL_CANDIDATES:
            if key in data:
                image = data[key]
                break
        if not image and data:
            image = next(iter(data.values()))   # fall back to first data field

        # Take the most recent annotation
        annos = task.get("annotations") or task.get("completions") or []
        if not annos:
            continue
        results = annos[-1].get("result", [])

        data = task.get("data", {})
        row = {
            "image": image,
            "severity": None,
            "dark_circles": 0,
            "quality_reject": 0,
            "makeup_suspected": 0,
            "annotation_confidence": data.get("annotation_confidence", data.get("confidence", "")),
            "annotator_id": data.get("annotator_id", ""),
            "subject_id": data.get("subject_id", ""),
            "source_dataset": data.get("source_dataset", ""),
            "source_image_id": data.get("source_image_id", data.get("source_image", "")),
            "license_status": data.get("license_status", ""),
            "consent_status": data.get("consent_status", ""),
        }
        for r in results:
            name = (r.get("from_name") or "").lower()
            vals = r.get("value", {}).get("choices", [])
            val  = vals[0] if vals else None
            if val is None:
                continue
            if any(c in name for c in ["sever", "grade"]):
                row["severity"] = val
            elif "dark" in name:
                row["dark_circles"] = val
            elif any(c in name for c in ["reject", "quality", "unusable"]):
                row["quality_reject"] = val
            elif "makeup" in name:
                row["makeup_suspected"] = val
            elif "confidence" in name:
                row["annotation_confidence"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Flatten Label Studio export into training CSV")
    parser.add_argument("--input",       required=True, help="Label Studio export (.csv or .json)")
    parser.add_argument("--output",      required=True, help="Output flat CSV path")
    parser.add_argument("--subject-map", default="",
                        help="Optional CSV mapping filename_stem → subject_id")
    parser.add_argument("--crops-prefix", default="",
                        help="Extra prefix to prepend to relative image paths (rarely needed)")
    parser.add_argument("--ls-root", default="data",
                        help="Repo-relative location of the Label Studio document root "
                             "(default: data). Prepended to '?d=' paths so training "
                             "resolves them. Use '' if your doc root is the repo root.")
    parser.add_argument("--source-dataset", default="",
                        help="Dataset/collection name to stamp when absent from input")
    parser.add_argument("--license-status", default="",
                        help="License bucket to stamp when absent from input, e.g. cc_by_4_0")
    parser.add_argument("--consent-status", default="unspecified",
                        help="Consent/release bucket to stamp when absent from input")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser()
    if not in_path.exists():
        print(f"❌ Input not found: {in_path}")
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────
    if in_path.suffix.lower() == ".json":
        df = load_labelstudio_json(in_path)
        img_col, sev_col, dark_col, rej_col = "image", "severity", "dark_circles", "quality_reject"
        makeup_col = "makeup_suspected"
        conf_col = "annotation_confidence"
        annotator_col = "annotator_id"
        subject_col = "subject_id"
        source_dataset_col = "source_dataset"
        source_image_col = "source_image_id"
        license_col = "license_status"
        consent_col = "consent_status"
    else:
        df = pd.read_csv(in_path)
        img_col  = _find_col(df, IMAGE_COL_CANDIDATES)
        sev_col  = _find_col(df, SEVERITY_COL_CANDIDATES)
        dark_col = _find_col(df, DARK_COL_CANDIDATES)
        rej_col  = _find_col(df, REJECT_COL_CANDIDATES)
        makeup_col = _find_col(df, MAKEUP_COL_CANDIDATES)
        conf_col = _find_col(df, CONF_COL_CANDIDATES)
        annotator_col = _find_col(df, ANNOTATOR_COL_CANDIDATES)
        subject_col = _find_col(df, SUBJECT_COL_CANDIDATES)
        source_dataset_col = _find_col(df, SOURCE_DATASET_COL_CANDIDATES)
        source_image_col = _find_col(df, SOURCE_IMAGE_COL_CANDIDATES)
        license_col = _find_col(df, LICENSE_COL_CANDIDATES)
        consent_col = _find_col(df, CONSENT_COL_CANDIDATES)

    if not img_col or not sev_col:
        print(f"❌ Could not auto-detect image/severity columns.")
        print(f"   Found columns: {list(df.columns)}")
        print(f"   Rename your columns or edit the candidate lists at the top of this script.")
        sys.exit(1)

    print(f"Detected columns: image='{img_col}'  severity='{sev_col}'  "
          f"dark_circles='{dark_col or '(absent→0)'}'  reject='{rej_col or '(absent→0)'}'")

    # ── Build the flat frame ──────────────────────────────────────────────
    source_dataset_missing = not source_dataset_col and not args.source_dataset
    license_missing = not license_col and not args.license_status
    if source_dataset_missing or license_missing:
        print("Missing required provenance.")
        if source_dataset_missing:
            print("   Provide --source-dataset or include a source_dataset column.")
        if license_missing:
            print("   Provide --license-status or include a license_status column.")
        sys.exit(1)

    out = pd.DataFrame()
    out["image_path"] = df[img_col].apply(lambda r: _clean_image_path(r, args.ls_root))
    if args.crops_prefix:
        out["image_path"] = out["image_path"].apply(
            lambda p: str(Path(args.crops_prefix) / p) if not Path(p).is_absolute() else p
        )

    out["severity"]       = df[sev_col].apply(_severity_to_int)
    out["dark_circles"]   = df[dark_col].apply(_binary) if dark_col else 0
    out["quality_reject"] = df[rej_col].apply(_binary)  if rej_col  else 0
    out["makeup_suspected"] = df[makeup_col].apply(_binary) if makeup_col else 0
    out["presence"]       = (out["severity"] > 0).astype(int)
    out["eye"]            = out["image_path"].apply(_eye_from_filename)

    # ── Subject IDs ───────────────────────────────────────────────────────
    if args.subject_map:
        smap = pd.read_csv(args.subject_map)
        mapping = dict(zip(smap["filename_stem"].astype(str), smap["subject_id"].astype(str)))
        out["subject_id"] = out["image_path"].apply(
            lambda p: mapping.get(_subject_from_filename(p), _subject_from_filename(p))
        )
        print(f"Applied subject map: {len(mapping)} entries")
    elif subject_col:
        subject_values = _series_or_default(df, subject_col)
        out["subject_id"] = subject_values.where(
            subject_values.str.strip() != "",
            out["image_path"].apply(_subject_from_filename),
        )
    else:
        out["subject_id"] = out["image_path"].apply(_subject_from_filename)
        print("⚠️  No --subject-map provided. Using filename stem as subject_id.")
        print("   This assumes ONE PHOTO PER PERSON. If any person appears in")
        print("   multiple photos, your splits WILL leak — provide a subject map.")

    # ── Drop unparseable severities ───────────────────────────────────────
    out["source_dataset"] = _series_or_default(
        df, source_dataset_col, args.source_dataset
    ).replace("", args.source_dataset)
    source_image_values = _series_or_default(df, source_image_col)
    out["source_image_id"] = source_image_values.where(
        source_image_values.str.strip() != "",
        out["image_path"].apply(_subject_from_filename),
    )
    out["license_status"] = _series_or_default(
        df, license_col, args.license_status
    ).replace("", args.license_status)
    out["consent_status"] = _series_or_default(
        df, consent_col, args.consent_status
    ).replace("", args.consent_status)
    out["annotation_confidence"] = _series_or_default(
        df, conf_col, "medium"
    ).replace("", "medium")
    out["annotator_id"] = _series_or_default(df, annotator_col)

    empty_required = {
        col: int((out[col].isna() | (out[col].astype(str).str.strip() == "")).sum())
        for col in ["subject_id", "source_dataset", "license_status"]
    }
    empty_required = {k: v for k, v in empty_required.items() if v}
    if empty_required:
        print(f"Required provenance fields are blank in output rows: {empty_required}")
        sys.exit(1)

    bad = out[out["severity"] < 0]
    if not bad.empty:
        print(f"⚠️  Dropping {len(bad)} rows with unparseable severity values "
              f"(examples: {df[sev_col].iloc[bad.index[:3]].tolist()})")
        out = out[out["severity"] >= 0].reset_index(drop=True)

    # ── Save + summary ────────────────────────────────────────────────────
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"\n✅ Wrote {len(out)} rows → {out_path}")
    print(f"\nSeverity distribution:")
    for grade, cnt in out["severity"].value_counts().sort_index().items():
        print(f"  grade {grade}: {cnt:>5}  ({cnt/len(out)*100:.1f}%)")
    print(f"\nDark circles positive: {out['dark_circles'].sum()} ({out['dark_circles'].mean()*100:.1f}%)")
    print(f"Quality rejects:       {out['quality_reject'].sum()}")
    print(f"Unique subjects:       {out['subject_id'].nunique()}")
    print(f"Source datasets:       {sorted(out['source_dataset'].dropna().unique().tolist())}")
    print(f"License statuses:      {sorted(out['license_status'].dropna().unique().tolist())}")
    print(f"\nNext step:")
    print(f"  python -m src.data.splits --annotations {out_path} --output data/splits")


if __name__ == "__main__":
    main()
