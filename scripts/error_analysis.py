#!/usr/bin/env python3
"""
Error analysis for the DermaLens eye-bag model (Day 6 — DO NOT SKIP).

What this does:
  1. Runs the trained model over a split.
  2. Saves a per-image predictions CSV (probability, predicted grade, error).
  3. Builds image grids of the WORST mistakes:
       - top-20 false positives (model said "bag", truth says "no bag")
       - top-20 false negatives (model missed a real bag)
       - largest severity errors (if the severity head is enabled)
  4. Prints a leakage sanity check (suspiciously high accuracy warning).

How to read the output grids:
  False positives that all share strong shadows → the model learned
  "shadow = bag". Fix: more shadow negatives, check augmentation.
  False negatives that are all mild grade-1 cases → labels may be inconsistent
  at the 0/1 boundary. Fix: refine rubric, re-annotate borderline images.

Usage:
    python scripts/error_analysis.py \
        --checkpoint experiments/baseline_binary/best.pt \
        --csv data/splits/val.csv \
        --out experiments/baseline_binary/error_analysis
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset       import EyeBagDataset
from src.data.augmentations import get_val_transforms
from src.models.multitask   import load_model_from_checkpoint
from src.models.ordinal_head import coral_logits_to_grade, coral_grade_confidence

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def run_predictions(model, dataset, device, batch_size=64) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    rows = []
    for batch in loader:
        images = batch["image"].to(device)
        out = model(images)

        presence_prob = torch.sigmoid(out["presence_logit"]).cpu().numpy()

        if "severity_logits" in out:
            pred_grade = coral_logits_to_grade(out["severity_logits"].cpu()).numpy()
            confidence = coral_grade_confidence(out["severity_logits"].cpu()).numpy()
        else:
            pred_grade = np.full(len(presence_prob), -1)
            confidence = np.abs(presence_prob - 0.5) * 2   # binary confidence proxy

        for i in range(len(presence_prob)):
            rows.append({
                "image_path":    batch["image_path"][i],
                "subject_id":    batch["subject_id"][i],
                "true_presence": float(batch["presence"][i]),
                "true_severity": int(batch["severity"][i]),
                "pred_presence_prob": float(presence_prob[i]),
                "pred_severity":      int(pred_grade[i]),
                "confidence":         float(confidence[i]),
            })

    df = pd.DataFrame(rows)
    df["presence_error"] = np.abs(df["true_presence"] - df["pred_presence_prob"])
    df["severity_error"] = np.where(
        df["pred_severity"] >= 0,
        np.abs(df["true_severity"] - df["pred_severity"]),
        np.nan,
    )
    return df


def save_image_grid(df_subset: pd.DataFrame, title: str, out_path: Path, n_cols=5):
    """Save a grid of the actual crop images with predictions annotated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(df_subset)
    if n == 0:
        return
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 2.4))
    axes = np.atleast_2d(axes)

    for ax in axes.flat:
        ax.axis("off")

    for i, (_, row) in enumerate(df_subset.iterrows()):
        ax = axes.flat[i]
        try:
            img = Image.open(row["image_path"])
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "load failed", ha="center")
        ax.set_title(
            f"true={row['true_severity']}  pred_p={row['pred_presence_prob']:.2f}"
            + (f"  pred_g={row['pred_severity']}" if row["pred_severity"] >= 0 else ""),
            fontsize=7,
        )

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved grid → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Error analysis for trained model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv",        required=True, help="Split CSV to analyse (usually val.csv)")
    parser.add_argument("--out",        default="",    help="Output directory")
    parser.add_argument("--crops-dir",  default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out or (Path(args.checkpoint).parent / "error_analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model   = load_model_from_checkpoint(args.checkpoint, device)
    dataset = EyeBagDataset(args.csv, transform=get_val_transforms(), crops_dir=args.crops_dir)
    df      = run_predictions(model, dataset, device)

    # ── Save full predictions ────────────────────────────────────────────
    pred_path = out_dir / "predictions.csv"
    df.to_csv(pred_path, index=False)
    logger.info(f"Predictions saved → {pred_path}")

    # ── Leakage sanity check ─────────────────────────────────────────────
    acc = ((df["pred_presence_prob"] > 0.5) == (df["true_presence"] > 0.5)).mean()
    logger.info(f"Presence accuracy on this split: {acc*100:.1f}%")
    if acc > 0.95 and len(df) > 100:
        logger.warning(
            "⚠️  Accuracy above 95%% on the first run is SUSPICIOUS. Check for:\n"
            "    1. Duplicate images across train/val (run audit_dataset.py)\n"
            "    2. Same subject in both splits (check subject_id columns)\n"
            "    3. A trivially separable dataset (e.g. all positives from one source)"
        )

    # ── Worst mistakes grids ──────────────────────────────────────────────
    fp = df[(df["true_presence"] == 0)].nlargest(20, "pred_presence_prob")
    fn = df[(df["true_presence"] == 1)].nsmallest(20, "pred_presence_prob")
    save_image_grid(fp, "Worst FALSE POSITIVES (model sees bags that aren't there)",
                    out_dir / "false_positives.png")
    save_image_grid(fn, "Worst FALSE NEGATIVES (model missed real bags)",
                    out_dir / "false_negatives.png")

    if df["severity_error"].notna().any():
        worst_sev = df.nlargest(20, "severity_error")
        save_image_grid(worst_sev, "Largest SEVERITY errors",
                        out_dir / "severity_errors.png")

    # ── Console summary ───────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("ERROR ANALYSIS SUMMARY")
    print(f"{'='*55}")
    print(f"Samples analysed   : {len(df)}")
    print(f"Presence accuracy  : {acc*100:.1f}%")
    print(f"False positives    : {((df.pred_presence_prob>0.5)&(df.true_presence==0)).sum()}")
    print(f"False negatives    : {((df.pred_presence_prob<=0.5)&(df.true_presence==1)).sum()}")
    if df["severity_error"].notna().any():
        print(f"Severity MAE       : {df.severity_error.mean():.2f}")
        print(f"Within-one-grade   : {(df.severity_error<=1).mean()*100:.1f}%")
    print(f"\nNow OPEN the grids in {out_dir}/ and LOOK at the images.")
    print("Patterns to hunt for:")
    print("  - FPs all have shadows/dark circles → model confuses shadow with bags")
    print("  - FNs all grade-1 → 0/1 label boundary is inconsistent, refine rubric")
    print("  - FPs from one data source → source bias, check dataset mixing")


if __name__ == "__main__":
    main()
