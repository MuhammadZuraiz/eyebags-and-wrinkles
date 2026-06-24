"""Verify subject-level splitting never leaks subjects across partitions."""
import pandas as pd
import numpy as np
import pytest

from src.data.splits import split_annotations


def _fake_annotations(n_subjects=100, photos_per_subject=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        sev = int(rng.integers(0, 5))
        for p in range(photos_per_subject):
            for eye in ["left", "right"]:
                rows.append({
                    "image_path":   f"crops/{eye}/subj{s:03d}_p{p}_{eye}.jpg",
                    "severity":     sev,
                    "dark_circles": int(rng.integers(0, 2)),
                    "subject_id":   f"subj{s:03d}",
                    "eye":          eye,
                })
    return pd.DataFrame(rows)


def test_no_subject_overlap():
    df = _fake_annotations()
    train, val, ti, te = split_annotations(df, random_seed=7)
    sets = {
        "train": set(train.subject_id), "val": set(val.subject_id),
        "test_int": set(ti.subject_id), "test_ext": set(te.subject_id),
    }
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = sets[names[i]] & sets[names[j]]
            assert not overlap, f"Subjects leak between {names[i]} and {names[j]}: {overlap}"


def test_all_rows_assigned():
    df = _fake_annotations()
    parts = split_annotations(df, random_seed=7)
    assert sum(len(p) for p in parts) == len(df)


def test_rough_proportions():
    df = _fake_annotations(n_subjects=200)
    train, val, ti, te = split_annotations(df, random_seed=1)
    n = len(df)
    assert 0.60 < len(train) / n < 0.80
    assert 0.05 < len(val)   / n < 0.18


def test_image_level_fallback_warns():
    df = _fake_annotations()
    df["subject_id"] = ""   # simulate missing subject IDs
    with pytest.warns(UserWarning):
        parts = split_annotations(df, random_seed=3)
    assert sum(len(p) for p in parts) == len(df)
