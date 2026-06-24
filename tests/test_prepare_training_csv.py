"""Tests for Label Studio export normalization and provenance preservation."""
import sys

import pandas as pd
import pytest

from scripts import prepare_training_csv


def test_prepare_training_csv_preserves_provenance(tmp_path, monkeypatch):
    src = tmp_path / "export.csv"
    out = tmp_path / "annotations.csv"
    pd.DataFrame({
        "eye_crop": ["crops/subj001_left.jpg"],
        "severity": ["2 - Moderate"],
        "dark_circles": ["yes"],
        "quality_reject": ["usable"],
        "makeup_suspected": ["no"],
        "subject_id": ["subj001"],
        "source_dataset": ["london_faces"],
        "source_image_id": ["subj001"],
        "license_status": ["cc_by_4_0"],
        "consent_status": ["documented_research_consent"],
        "annotation_confidence": ["high"],
        "annotator_id": ["ann_a"],
    }).to_csv(src, index=False)

    monkeypatch.setattr(sys, "argv", [
        "prepare_training_csv.py",
        "--input", str(src),
        "--output", str(out),
    ])

    prepare_training_csv.main()

    df = pd.read_csv(out)
    assert df.loc[0, "image_path"] == "crops/subj001_left.jpg"
    assert df.loc[0, "severity"] == 2
    assert df.loc[0, "dark_circles"] == 1
    assert df.loc[0, "quality_reject"] == 0
    assert df.loc[0, "source_dataset"] == "london_faces"
    assert df.loc[0, "source_image_id"] == "subj001"
    assert df.loc[0, "license_status"] == "cc_by_4_0"
    assert df.loc[0, "consent_status"] == "documented_research_consent"
    assert df.loc[0, "annotation_confidence"] == "high"
    assert df.loc[0, "annotator_id"] == "ann_a"


def test_prepare_training_csv_requires_source_and_license(tmp_path, monkeypatch):
    src = tmp_path / "export.csv"
    out = tmp_path / "annotations.csv"
    pd.DataFrame({
        "image_path": ["crops/subj001_left.jpg"],
        "severity": ["1 - Mild"],
        "dark_circles": ["no"],
    }).to_csv(src, index=False)

    monkeypatch.setattr(sys, "argv", [
        "prepare_training_csv.py",
        "--input", str(src),
        "--output", str(out),
    ])

    with pytest.raises(SystemExit):
        prepare_training_csv.main()
