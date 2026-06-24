"""Test the EyeBagDataset + sampler on a synthetic CSV with tiny images."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.dataset import EyeBagDataset
from src.data.sampler import build_balanced_sampler
from src.data.augmentations import get_val_transforms, get_train_transforms


@pytest.fixture
def tiny_dataset(tmp_path):
    """Create 20 fake crops + CSV."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        img = Image.fromarray(rng.integers(0, 255, (160, 256, 3), dtype=np.uint8))
        p = tmp_path / f"crop_{i:02d}.jpg"
        img.save(p)
        rows.append({
            "image_path":   str(p),
            "severity":     int(rng.integers(0, 5)),
            "dark_circles": int(rng.integers(0, 2)),
            "subject_id":   f"s{i//2}",   # 2 crops per subject
            "source_dataset": "unit_test",
            "source_image_id": f"s{i//2}",
            "license_status": "test_only",
            "consent_status": "synthetic",
            "makeup_suspected": 0,
            "annotation_confidence": "medium",
            "annotator_id": "pytest",
        })
    csv = tmp_path / "ann.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_dataset_loads(tiny_dataset):
    ds = EyeBagDataset(str(tiny_dataset), transform=get_val_transforms())
    assert len(ds) == 20
    sample = ds[0]
    assert sample["image"].shape == (3, 160, 256)
    assert sample["severity"].item() in range(5)
    assert sample["presence"].item() in (0.0, 1.0)
    assert sample["source_dataset"] == "unit_test"
    assert sample["license_status"] == "test_only"


def test_presence_derived_from_severity(tiny_dataset):
    ds = EyeBagDataset(str(tiny_dataset))
    for i in range(len(ds)):
        row = ds.df.iloc[i]
        assert bool(row["presence"]) == (row["severity"] > 0)


def test_train_transforms_run(tiny_dataset):
    ds = EyeBagDataset(str(tiny_dataset), transform=get_train_transforms())
    sample = ds[0]
    assert sample["image"].shape == (3, 160, 256)
    assert sample["image"].dtype.is_floating_point


def test_balanced_sampler(tiny_dataset):
    ds = EyeBagDataset(str(tiny_dataset))
    sampler = build_balanced_sampler(ds)
    indices = list(iter(sampler))
    assert len(indices) == len(ds)
    assert all(0 <= i < len(ds) for i in indices)


def test_missing_columns_raise(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"image_path": ["x.jpg"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        EyeBagDataset(str(bad))


def test_missing_provenance_values_raise(tmp_path):
    img = Image.fromarray(np.zeros((160, 256, 3), dtype=np.uint8))
    p = tmp_path / "crop.jpg"
    img.save(p)
    bad = tmp_path / "bad_provenance.csv"
    pd.DataFrame({
        "image_path": [str(p)],
        "severity": [1],
        "dark_circles": [0],
        "subject_id": [""],
        "source_dataset": ["unit_test"],
        "license_status": ["test_only"],
    }).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="non-empty provenance"):
        EyeBagDataset(str(bad))


def test_missing_image_file_raises_at_load(tmp_path):
    """A CSV row pointing at a nonexistent crop must fail at dataset
    construction — never silently train on a blank placeholder."""
    csv = tmp_path / "missing_img.csv"
    pd.DataFrame({
        "image_path": [str(tmp_path / "does_not_exist.jpg")],
        "severity": [1],
        "dark_circles": [0],
        "subject_id": ["s0"],
        "source_dataset": ["unit_test"],
        "license_status": ["test_only"],
    }).to_csv(csv, index=False)
    with pytest.raises(FileNotFoundError, match="do not exist"):
        EyeBagDataset(str(csv))


def test_corrupt_image_raises_in_getitem(tmp_path):
    """A file that exists but cannot be decoded must raise, not degrade."""
    fake = tmp_path / "corrupt.jpg"
    fake.write_bytes(b"this is not a jpeg")
    csv = tmp_path / "corrupt_img.csv"
    pd.DataFrame({
        "image_path": [str(fake)],
        "severity": [1],
        "dark_circles": [0],
        "subject_id": ["s0"],
        "source_dataset": ["unit_test"],
        "license_status": ["test_only"],
    }).to_csv(csv, index=False)
    ds = EyeBagDataset(str(csv))
    with pytest.raises(RuntimeError, match="Failed to load"):
        _ = ds[0]


def test_allow_missing_images_escape_hatch(tmp_path):
    """The smoke-only escape hatch keeps the old blank-image behavior."""
    csv = tmp_path / "missing_img.csv"
    pd.DataFrame({
        "image_path": [str(tmp_path / "does_not_exist.jpg")],
        "severity": [1],
        "dark_circles": [0],
        "subject_id": ["s0"],
        "source_dataset": ["unit_test"],
        "license_status": ["test_only"],
    }).to_csv(csv, index=False)
    ds = EyeBagDataset(str(csv), allow_missing_images=True)
    sample = ds[0]
    assert sample["severity"].item() == 1
