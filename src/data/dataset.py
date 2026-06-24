#!/usr/bin/env python3
"""
PyTorch Dataset for DermaLens under-eye crops.

DATA MODEL — read this first:
  ONE ROW = ONE CROP = ONE EYE.
  The model analyses a single under-eye crop at a time, so the training CSV is
  "flat": every row is one 256x160 crop with its own severity grade.
  A face photo therefore produces TWO rows (left crop + right crop).

  Use scripts/prepare_training_csv.py to convert your Label Studio export
  (which is per-face) into this flat per-crop format.

Required CSV columns:
    image_path     - path to the crop image (256x160 jpg)
    severity       - severity grade for THIS crop's eye (0, 1, 2, 3, or 4)
    dark_circles   - 1 if dark circles visible under this eye, 0 if not
    subject_id     - participant/source identity. CRITICAL for leak-free splits.
    source_dataset - dataset or collection this row came from
    license_status - reviewed license bucket for this row

Optional columns (defaults applied if absent):
    presence       - 1 if eye bag visible (derived from severity > 0 if absent)
    eye            - "left" or "right" (metadata only; orientation is already
                     normalised by batch_crop.py, so the model doesn't need it)
    quality_reject - 1 if annotator marked the image unusable (row is dropped)
    source_image_id - source photo/session ID before per-eye cropping
    consent_status - consent/release bucket for this row
    makeup_suspected - 1 if makeup is suspected, 0 otherwise
    annotation_confidence - annotator confidence: low / medium / high
    annotator_id    - annotator identifier for audit/adjudication
    mst_shade      - Monk Skin Tone scale 1-10 (0 = unknown) for fairness audits
    age_band       - e.g. "25_34"
    lighting       - e.g. "warm_indoor"

Minimal example CSV:
    image_path,severity,dark_circles,subject_id,source_dataset,license_status
    data/crops/left/img001_left.jpg,2,1,subj_001,london_faces,cc_by_4_0
    data/crops/right/img001_right.jpg,2,1,subj_001,london_faces,cc_by_4_0
    data/crops/left/img002_left.jpg,0,0,subj_002,london_faces,cc_by_4_0

Usage:
    from torch.utils.data import DataLoader
    from src.data.dataset import EyeBagDataset
    from src.data.augmentations import get_train_transforms, get_val_transforms

    train_ds = EyeBagDataset("data/splits/train.csv", transform=get_train_transforms())
    val_ds   = EyeBagDataset("data/splits/val.csv",   transform=get_val_transforms())

    # Each batch is a dict:
    #   batch["image"]        -> (B, 3, 160, 256) float tensor (ImageNet-normalised)
    #   batch["severity"]     -> (B,) int64, values 0-4
    #   batch["presence"]     -> (B,) float32, 0.0 or 1.0
    #   batch["dark_circles"] -> (B,) float32, 0.0 or 1.0
    #   batch["subject_id"]   -> list of strings
    #   batch["source_dataset"] -> list of strings
    #   batch["license_status"] -> list of strings
    #   batch["mst_shade"]    -> (B,) int64 (0 = unknown)
    #   batch["image_path"]   -> list of path strings (for error analysis)
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

SEVERITY_GRADES = [0, 1, 2, 3, 4]
NUM_GRADES      = len(SEVERITY_GRADES)

REQUIRED_COLS = {
    "image_path",
    "severity",
    "dark_circles",
    "subject_id",
    "source_dataset",
    "license_status",
}

OPTIONAL_DEFAULTS = {
    "presence":              None,      # derived from severity if absent
    "eye":                   "",
    "quality_reject":        0,
    "source_image_id":       "",
    "consent_status":        "unspecified",
    "makeup_suspected":      0,
    "annotation_confidence": "medium",
    "annotator_id":          "",
    "mst_shade":             0,
    "age_band":              "",
    "lighting":              "",
}

NONEMPTY_REQUIRED_COLS = {"subject_id", "source_dataset", "license_status"}


class EyeBagDataset(Dataset):
    """
    Loads per-crop annotation rows from a flat CSV.

    Args:
        csv_path:             Path to the flat annotation CSV (format above).
        transform:            Callable applied to the PIL image.
                              get_train_transforms() / get_val_transforms().
        skip_quality_rejects: Drop rows where quality_reject == 1 (default True).
        crops_dir:            Optional prefix prepended to relative image_path values.
        allow_missing_images: Escape hatch for synthetic smoke runs only. When
                              False (default), a CSV row pointing at a missing or
                              unreadable image raises instead of silently training
                              on a blank placeholder.
    """

    def __init__(
        self,
        csv_path:             str,
        transform:            Optional[Callable] = None,
        skip_quality_rejects: bool = True,
        crops_dir:            Optional[str] = None,
        allow_missing_images: bool = False,
    ):
        self.transform = transform
        self.crops_dir = Path(crops_dir) if crops_dir else None
        self.allow_missing_images = allow_missing_images

        df = pd.read_csv(csv_path)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}\n"
                f"Found columns: {list(df.columns)}\n"
                f"Did you run scripts/prepare_training_csv.py on your Label Studio export?"
            )

        for col, default in OPTIONAL_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default

        # Derive presence from severity where not annotated
        mask = df["presence"].isna()
        df.loc[mask, "presence"] = (df.loc[mask, "severity"] > 0).astype(int)

        if skip_quality_rejects:
            n_before = len(df)
            df = df[df["quality_reject"].fillna(0).astype(int) == 0].reset_index(drop=True)
            removed = n_before - len(df)
            if removed:
                logger.info(f"Removed {removed} quality-rejected rows from {csv_path}")

        empty_required = {}
        for col in NONEMPTY_REQUIRED_COLS:
            empty_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
            if empty_mask.any():
                empty_required[col] = int(empty_mask.sum())
        if empty_required:
            raise ValueError(
                "Training rows must include non-empty provenance fields: "
                f"{empty_required}. Add these via scripts/prepare_training_csv.py "
                "or fix the source annotation CSV."
            )

        invalid = df[~df["severity"].isin(SEVERITY_GRADES)]
        if not invalid.empty:
            raise ValueError(
                f"{len(invalid)} rows have severity outside {SEVERITY_GRADES}. "
                f"First bad rows: {invalid.head(3).to_dict('records')}"
            )

        self.df = df.reset_index(drop=True)

        if not self.allow_missing_images:
            missing_files = [
                str(p) for p in (self._resolve_path(v) for v in self.df["image_path"])
                if not p.is_file()
            ]
            if missing_files:
                preview = "\n  ".join(missing_files[:5])
                raise FileNotFoundError(
                    f"{len(missing_files)} of {len(self.df)} crop images listed in "
                    f"{csv_path} do not exist. First missing:\n  {preview}\n"
                    "Fix the CSV/crops (or pass allow_missing_images=True for "
                    "synthetic smoke runs only)."
                )

        counts = self.df["severity"].value_counts().sort_index().to_dict()
        logger.info(f"Dataset loaded: {len(self.df)} crops from {csv_path}  grade counts: {counts}")

    def _resolve_path(self, value: str) -> Path:
        p = Path(value)
        if self.crops_dir is not None and not p.is_absolute():
            p = self.crops_dir / p
        return p

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]

        img_path = self._resolve_path(row["image_path"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            if not self.allow_missing_images:
                raise RuntimeError(
                    f"Failed to load crop image {img_path}: {exc}. "
                    "Training must not proceed on placeholder images."
                ) from exc
            logger.warning(f"Failed to load {img_path}: {exc}. Using blank image.")
            image = Image.new("RGB", (256, 160), (128, 128, 128))

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image":        image,
            "severity":     torch.tensor(int(row["severity"]),      dtype=torch.long),
            "presence":     torch.tensor(float(row["presence"]),    dtype=torch.float32),
            "dark_circles": torch.tensor(float(row["dark_circles"]),dtype=torch.float32),
            "subject_id":   str(row.get("subject_id", "") or ""),
            "eye":          str(row.get("eye", "") or ""),
            "source_dataset": str(row.get("source_dataset", "") or ""),
            "source_image_id": str(row.get("source_image_id", "") or ""),
            "license_status": str(row.get("license_status", "") or ""),
            "consent_status": str(row.get("consent_status", "") or ""),
            "makeup_suspected": torch.tensor(float(row.get("makeup_suspected", 0) or 0),
                                             dtype=torch.float32),
            "annotation_confidence": str(row.get("annotation_confidence", "") or ""),
            "annotator_id": str(row.get("annotator_id", "") or ""),
            "mst_shade":    int(row.get("mst_shade", 0) or 0),
            "age_band":     str(row.get("age_band", "") or ""),
            "lighting":     str(row.get("lighting", "") or ""),
            "image_path":   str(img_path),
        }

    # ── Convenience methods ──────────────────────────────────────────────

    def grade_distribution(self) -> Dict[int, int]:
        return dict(self.df["severity"].value_counts().sort_index())

    def subject_ids(self) -> List[str]:
        return self.df["subject_id"].dropna().unique().tolist()

    def mst_distribution(self) -> Dict[int, int]:
        return dict(self.df["mst_shade"].value_counts().sort_index())
