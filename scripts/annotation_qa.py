#!/usr/bin/env python3
"""
Annotation self-consistency check for the calibration batch.

WHY: a single-annotator dataset's quality ceiling is the annotator's own
consistency — the model cannot be more reliable than the labels. Before
annotating the full set, the same ~80 crops are graded twice (reshuffled) and
this script measures how stable the grades are.

GATE (from the project plan):
    proceed when exact Cohen's kappa >= 0.6  OR  within-one agreement >= 0.8
    otherwise: tighten the rubric/atlas and redo a fresh calibration batch.

Inputs are two Label Studio exports (CSV or JSON) of the SAME tasks. Rows are
matched by the eye-crop image path/name.

Usage:
    python scripts/annotation_qa.py --pass-a exports/calib_a.csv --pass-b exports/calib_b.csv
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────────────────────────────────────
# Export parsing (works for both LS CSV and LS JSON exports)
# ──────────────────────────────────────────────────────────────────────────────

def _severity_to_int(value) -> int:
    """'2 - Moderate' -> 2; '3' -> 3; returns -1 when unparseable."""
    m = re.match(r"\s*(\d)", str(value))
    return int(m.group(1)) if m else -1


def _crop_key(value) -> str:
    """Normalize an eye-crop reference to its bare filename."""
    s = str(value)
    s = s.split("?d=")[-1]          # strip /data/local-files/?d= prefix
    return Path(s).name


def load_export(path: Path) -> pd.DataFrame:
    """Return DataFrame with columns: crop, severity, quality_reject."""
    rows = []
    if path.suffix.lower() == ".json":
        tasks = json.loads(path.read_text(encoding="utf-8"))
        for task in tasks:
            data = task.get("data", {})
            crop = data.get("eye_crop", "")
            severity, reject = None, ""
            for ann in task.get("annotations", []):
                for res in ann.get("result", []):
                    name = res.get("from_name", "")
                    vals = res.get("value", {}).get("choices", [])
                    if not vals:
                        continue
                    if name == "severity":
                        severity = vals[0]
                    elif name == "quality_reject":
                        reject = vals[0]
            if crop and severity is not None:
                rows.append({"crop": _crop_key(crop),
                             "severity": _severity_to_int(severity),
                             "quality_reject": reject})
    else:
        df = pd.read_csv(path)
        crop_col = next((c for c in df.columns if c in ("eye_crop", "image", "crop")), None)
        sev_col  = next((c for c in df.columns if "severity" in c.lower()), None)
        rej_col  = next((c for c in df.columns if "quality" in c.lower()), None)
        if crop_col is None or sev_col is None:
            raise SystemExit(
                f"Could not find eye_crop/severity columns in {path}. "
                f"Columns: {list(df.columns)}"
            )
        for _, r in df.iterrows():
            rows.append({"crop": _crop_key(r[crop_col]),
                         "severity": _severity_to_int(r[sev_col]),
                         "quality_reject": str(r[rej_col]) if rej_col else ""})
    out = pd.DataFrame(rows)
    return out[out["severity"] >= 0].drop_duplicates(subset="crop", keep="last")


# ──────────────────────────────────────────────────────────────────────────────
# Agreement metrics
# ──────────────────────────────────────────────────────────────────────────────

def cohens_kappa(a: np.ndarray, b: np.ndarray, weights: str = "none",
                 n_classes: int = 5) -> float:
    """Cohen's kappa; weights: 'none' (exact) or 'quadratic'."""
    conf = np.zeros((n_classes, n_classes), dtype=float)
    for x, y in zip(a, b):
        conf[x, y] += 1
    n = conf.sum()
    if n == 0:
        return float("nan")
    if weights == "quadratic":
        idx = np.arange(n_classes)
        w = ((idx[:, None] - idx[None, :]) ** 2) / (n_classes - 1) ** 2
    else:
        w = 1.0 - np.eye(n_classes)
    row = conf.sum(axis=1, keepdims=True)
    col = conf.sum(axis=0, keepdims=True)
    expected = row @ col / n
    observed_disagreement = (w * conf).sum() / n
    expected_disagreement = (w * expected).sum() / n
    if expected_disagreement == 0:
        return 1.0
    return 1.0 - observed_disagreement / expected_disagreement


def main():
    parser = argparse.ArgumentParser(description="Two-pass annotation agreement")
    parser.add_argument("--pass-a", required=True, help="First-pass LS export (csv/json)")
    parser.add_argument("--pass-b", required=True, help="Second-pass LS export (csv/json)")
    args = parser.parse_args()

    a = load_export(Path(args.pass_a))
    b = load_export(Path(args.pass_b))
    merged = a.merge(b, on="crop", suffixes=("_a", "_b"))
    if merged.empty:
        raise SystemExit("No overlapping crops between the two exports — "
                         "are these the same calibration batch?")

    sa = merged["severity_a"].to_numpy()
    sb = merged["severity_b"].to_numpy()
    n = len(merged)

    exact      = float((sa == sb).mean())
    within_one = float((np.abs(sa - sb) <= 1).mean())
    kappa      = cohens_kappa(sa, sb, "none")
    qwk        = cohens_kappa(sa, sb, "quadratic")

    print(f"Calibration agreement on {n} crops")
    print(f"  exact agreement     : {exact:.3f}")
    print(f"  within-one agreement: {within_one:.3f}")
    print(f"  Cohen's kappa       : {kappa:.3f}")
    print(f"  quadratic-wtd kappa : {qwk:.3f}")

    print("\nConfusion matrix (rows = pass A, cols = pass B):")
    conf = pd.crosstab(pd.Series(sa, name="A"), pd.Series(sb, name="B"))
    conf = conf.reindex(index=range(5), columns=range(5), fill_value=0)
    print(conf.to_string())

    # Which grade boundary is noisy?
    off = merged[sa != sb]
    if not off.empty:
        pairs = off.apply(
            lambda r: f"{min(r.severity_a, r.severity_b)}/"
                      f"{max(r.severity_a, r.severity_b)}", axis=1)
        print("\nDisagreements by boundary:")
        print(pairs.value_counts().to_string())
        big = off[np.abs(off["severity_a"] - off["severity_b"]) >= 2]
        if not big.empty:
            print(f"\n{len(big)} crops differ by >= 2 grades — REVIEW these "
                  f"before continuing:")
            for crop in big["crop"].head(10):
                print(f"  {crop}")

    passed = kappa >= 0.6 or within_one >= 0.8
    print("\n" + ("GATE PASSED: proceed to full annotation." if passed else
                  "GATE FAILED: tighten the rubric (docs/grade_atlas.md), add "
                  "tiebreak rules for the noisy boundary above, and redo a "
                  "fresh 50-crop calibration batch."))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
