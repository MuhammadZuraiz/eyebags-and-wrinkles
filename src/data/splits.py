#!/usr/bin/env python3
"""
Participant-level dataset splitting for DermaLens.

WHY THIS MATTERS (read this carefully):
  If the same person has multiple photos in your dataset and those photos end
  up in BOTH training and test, your model will appear to perform much better
  than it actually is on real-world data. It is not learning "eye bags" — it
  is partially memorising that person's face.

  This is called DATA LEAKAGE and it is the most common hidden mistake in
  face-image ML projects.

  The fix: split by SUBJECT ID, not by image. Every image from one person
  must stay in the same partition.

  If your images don't have subject IDs (e.g. you downloaded a public dataset
  with no participant metadata), this script will do a random image split with
  a warning. That is still better than nothing, but note the limitation.

Usage:
    # Recommended: split using subject IDs from the annotation CSV
    python scripts/create_subject_splits.py \
        --annotations data/annotations/all_annotations.csv \
        --output      data/splits \
        --train 0.70 --val 0.10 --test-internal 0.10 --test-external 0.10

    Output:
        data/splits/train.csv
        data/splits/val.csv
        data/splits/test_internal.csv
        data/splits/test_external.csv  ← FREEZE THIS. Never tune on it.
        data/splits/split_report.txt

    You can also call the split_annotations() function directly from Python.
"""

import hashlib
import logging
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────────────────────────────────────

def split_annotations(
    df:              pd.DataFrame,
    train_frac:      float = 0.70,
    val_frac:        float = 0.10,
    test_int_frac:   float = 0.10,
    test_ext_frac:   float = 0.10,
    subject_col:     str   = "subject_id",
    stratify_col:    Optional[str] = "severity",  # Try to balance grades across splits
    random_seed:     int   = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame of annotations into train / val / test_internal / test_external.

    The split is done at the SUBJECT level (all rows sharing a subject_id go
    to the same partition). Stratification is applied at the subject level
    to keep the severity distribution roughly balanced.

    Args:
        df:            Full annotation DataFrame from your annotations CSV.
        train_frac:    Fraction of subjects in the training set.
        val_frac:      Fraction of subjects in the validation set.
        test_int_frac: Fraction for internal test set.
        test_ext_frac: Fraction for external test set.
        subject_col:   Column name containing participant / subject IDs.
        stratify_col:  Column to use when stratifying subjects across splits.
                       Should be a label that represents the severity of that subject's images.
        random_seed:   For reproducibility.

    Returns:
        (train_df, val_df, test_int_df, test_ext_df)
        Four DataFrames — subsets of the original df.
    """
    assert abs(train_frac + val_frac + test_int_frac + test_ext_frac - 1.0) < 1e-6, \
        f"Fractions must sum to 1.0, got {train_frac+val_frac+test_int_frac+test_ext_frac:.4f}"

    rng = np.random.default_rng(random_seed)

    # ── Check if we have subject IDs ──────────────────────────────────────
    has_subjects = (
        subject_col in df.columns
        and df[subject_col].notna().any()
        and (df[subject_col] != "").any()
    )

    if not has_subjects:
        warnings.warn(
            f"Column '{subject_col}' is empty or absent. Falling back to IMAGE-LEVEL split. "
            f"This is WEAKER than a subject-level split — if the same person appears in multiple "
            f"images, those images may end up in different partitions. "
            f"For production models, always collect and record participant IDs.",
            stacklevel=2,
        )
        return _image_level_split(df, train_frac, val_frac, test_int_frac, test_ext_frac, rng)

    # ── Subject-level split ───────────────────────────────────────────────
    subjects = df[subject_col].unique().tolist()
    n_subjects = len(subjects)
    logger.info(f"Splitting {n_subjects} subjects ({len(df)} images)")

    # Build a per-subject representative grade (mode of their severity labels)
    if stratify_col in df.columns:
        subject_grade = (
            df.groupby(subject_col)[stratify_col]
            .apply(lambda x: x.mode()[0])
            .reset_index()
        )
        subject_grade.columns = [subject_col, "_strat_grade"]
    else:
        subject_grade = pd.DataFrame({
            subject_col:    subjects,
            "_strat_grade": [0] * n_subjects,
        })

    # Stratified shuffle: group subjects by grade, then round-robin assign
    subject_grade = subject_grade.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    grade_groups  = subject_grade.groupby("_strat_grade")[subject_col].apply(list).to_dict()

    train_subs, val_subs, test_int_subs, test_ext_subs = [], [], [], []

    for grade, subs in sorted(grade_groups.items()):
        rng.shuffle(subs)
        n    = len(subs)
        i1   = max(1, round(n * train_frac))
        i2   = max(i1 + 1, round(n * (train_frac + val_frac)))
        i3   = max(i2 + 1, round(n * (train_frac + val_frac + test_int_frac)))

        train_subs   .extend(subs[:i1])
        val_subs     .extend(subs[i1:i2])
        test_int_subs.extend(subs[i2:i3])
        test_ext_subs.extend(subs[i3:])

    # Map subjects back to rows
    train_df    = df[df[subject_col].isin(set(train_subs   ))].copy()
    val_df      = df[df[subject_col].isin(set(val_subs     ))].copy()
    test_int_df = df[df[subject_col].isin(set(test_int_subs))].copy()
    test_ext_df = df[df[subject_col].isin(set(test_ext_subs))].copy()

    _log_split_stats(train_df, val_df, test_int_df, test_ext_df, subject_col)
    return train_df, val_df, test_int_df, test_ext_df


def _image_level_split(df, train_frac, val_frac, test_int_frac, test_ext_frac, rng):
    """Fallback: random image-level split (no subject IDs available)."""
    idx    = rng.permutation(len(df))
    n      = len(idx)
    i1     = round(n * train_frac)
    i2     = round(n * (train_frac + val_frac))
    i3     = round(n * (train_frac + val_frac + test_int_frac))

    return (
        df.iloc[idx[:i1]].copy().reset_index(drop=True),
        df.iloc[idx[i1:i2]].copy().reset_index(drop=True),
        df.iloc[idx[i2:i3]].copy().reset_index(drop=True),
        df.iloc[idx[i3:]].copy().reset_index(drop=True),
    )


def _log_split_stats(train_df, val_df, test_int_df, test_ext_df, subject_col):
    total = sum(len(d) for d in [train_df, val_df, test_int_df, test_ext_df])
    for name, df in [
        ("train", train_df), ("val", val_df),
        ("test_int", test_int_df), ("test_ext", test_ext_df)
    ]:
        n_imgs  = len(df)
        n_subs  = df[subject_col].nunique() if subject_col in df.columns else "?"
        pct     = n_imgs / max(total, 1) * 100
        logger.info(f"  {name:<12}: {n_imgs:>5} images  {n_subs:>4} subjects  ({pct:.1f}%)")

    # Leakage check
    for (n1, d1), (n2, d2) in [
        (("train",    train_df),    ("val",      val_df)),
        (("train",    train_df),    ("test_int", test_int_df)),
        (("train",    train_df),    ("test_ext", test_ext_df)),
        (("val",      val_df),      ("test_int", test_int_df)),
        (("val",      val_df),      ("test_ext", test_ext_df)),
        (("test_int", test_int_df), ("test_ext", test_ext_df)),
    ]:
        if subject_col in d1.columns and subject_col in d2.columns:
            overlap = set(d1[subject_col]) & set(d2[subject_col])
            if overlap:
                logger.error(
                    f"⚠️  LEAKAGE: {len(overlap)} subjects in BOTH {n1} and {n2}! "
                    f"Check your splitting logic."
                )
            else:
                logger.info(f"  ✅ No subject overlap between {n1} and {n2}")


# ──────────────────────────────────────────────────────────────────────────────
# Script entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Create train/val/test splits from annotation CSV"
    )
    parser.add_argument("--annotations", required=True, help="Path to full annotation CSV")
    parser.add_argument("--output",      required=True, help="Output directory for split CSVs")
    parser.add_argument("--train",        type=float, default=0.70)
    parser.add_argument("--val",          type=float, default=0.10)
    parser.add_argument("--test-internal",type=float, default=0.10)
    parser.add_argument("--test-external",type=float, default=0.10)
    parser.add_argument("--subject-col",  default="subject_id")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.annotations)
    logger.info(f"Loaded {len(df)} rows from {args.annotations}")

    train_df, val_df, test_int_df, test_ext_df = split_annotations(
        df,
        train_frac      = args.train,
        val_frac        = args.val,
        test_int_frac   = args.test_internal,
        test_ext_frac   = args.test_external,
        subject_col     = args.subject_col,
        random_seed     = args.seed,
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    train_df   .to_csv(out / "train.csv",         index=False)
    val_df     .to_csv(out / "val.csv",            index=False)
    test_int_df.to_csv(out / "test_internal.csv",  index=False)
    test_ext_df.to_csv(out / "test_external.csv",  index=False)

    report = [
        "DermaLens Split Report",
        "=" * 40,
        f"Source: {args.annotations}",
        f"Total images: {len(df)}",
        f"",
        f"train.csv         : {len(train_df)} rows",
        f"val.csv           : {len(val_df)} rows",
        f"test_internal.csv : {len(test_int_df)} rows",
        f"test_external.csv : {len(test_ext_df)} rows  <-- FREEZE THIS",
        f"",
        f"WARNING: Do not look at test_external.csv results until the model",
        f"is completely finalised. Tuning on external test results invalidates",
        f"the evaluation.",
    ]
    (out / "split_report.txt").write_text("\n".join(report), encoding="utf-8")
    logger.info(f"Splits saved to {out}/")
    print("\n".join(report))


if __name__ == "__main__":
    main()
