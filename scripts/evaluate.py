#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on any split CSV — the milestone-gate tool.

Emits:
    - the full metric suite (presence: AUROC/AUPRC/F1/ECE/Brier;
      severity: MAE/exact/within-one/QWK; dark circles if the head exists)
    - subgroup reports by mst_shade, source_dataset and age_band
    - per-crop predictions CSV (feeds error review / re-annotation)
    - metrics JSON saved next to the checkpoint

PROTOCOL (do not bend it):
    iterate on val.csv -> confirm each milestone ONCE on test_internal.csv ->
    touch test_external.csv exactly once, for the final report.

Usage:
    python scripts/evaluate.py --checkpoint experiments/baseline_binary/best.pt ^
        --csv data/splits/val.csv
    python scripts/evaluate.py --checkpoint experiments/ordinal_severity/best.pt ^
        --csv data/splits/test_internal.csv --tag milestone2_internal
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset        import EyeBagDataset
from src.data.augmentations  import get_val_transforms
from src.models.multitask    import load_model_from_checkpoint
from src.models.ordinal_head import coral_logits_to_grade, coral_grade_confidence
from src.evaluation.metrics  import compute_metrics, subgroup_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

SUBGROUP_COLS = ["mst_shade", "source_dataset", "age_band"]


@torch.no_grad()
def collect_outputs(model, loader, device, tta=False):
    presence_logits, presence_targets = [], []
    severity_logits, severity_targets = [], []
    dc_logits, dc_targets = [], []
    paths = []

    def forward(images):
        out = model(images)
        if not tta:
            return out
        # Horizontal flip is label-safe: crops are orientation-normalised
        # (outer corner always left), so averaging logits over the flip is a
        # free, principled test-time boost applied identically to all splits.
        flip = model(torch.flip(images, dims=[3]))
        merged = {}
        for k in out:
            if k.endswith("logit") or k.endswith("logits"):
                merged[k] = 0.5 * (out[k] + flip[k])
            else:
                merged[k] = out[k]
        return merged

    for batch in loader:
        out = forward(batch["image"].to(device))
        presence_logits.append(out["presence_logit"].float().cpu())
        presence_targets.append(batch["presence"])
        if "severity_logits" in out:
            severity_logits.append(out["severity_logits"].float().cpu())
            severity_targets.append(batch["severity"])
        if "dark_circles_logit" in out:
            dc_logits.append(out["dark_circles_logit"].float().cpu())
            dc_targets.append(batch["dark_circles"])
        paths.extend(batch["image_path"])

    cat = lambda xs: torch.cat(xs) if xs else None
    return {
        "presence_logits":  cat(presence_logits),
        "presence_targets": cat(presence_targets),
        "severity_logits":  cat(severity_logits),
        "severity_targets": cat(severity_targets),
        "dc_logits":        cat(dc_logits),
        "dc_targets":       cat(dc_targets),
        "paths":            paths,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a split CSV")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv",        required=True, help="Split CSV to evaluate on")
    parser.add_argument("--crops-dir",  default=None,
                        help="Prefix for relative image paths (matches config data.crops_dir)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tta", action="store_true",
                        help="Horizontal-flip test-time augmentation (label-safe; "
                             "apply identically to val and the final test sets)")
    parser.add_argument("--tag",        default="",
                        help="Label for the output files (default: csv stem)")
    parser.add_argument("--out-dir",    default="",
                        help="Output dir (default: next to the checkpoint)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(args.checkpoint, device)

    ds = EyeBagDataset(args.csv, transform=get_val_transforms(),
                       crops_dir=args.crops_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logger.info(f"Evaluating {len(ds)} crops from {args.csv}")

    o = collect_outputs(model, loader, device, tta=args.tta)

    # ── Overall metrics ───────────────────────────────────────────────────
    metrics = compute_metrics(
        presence_logits  = o["presence_logits"],
        presence_targets = o["presence_targets"],
        severity_logits  = o["severity_logits"],
        severity_targets = o["severity_targets"],
        dc_logits        = o["dc_logits"],
        dc_targets       = o["dc_targets"],
    )

    print(f"\n=== {Path(args.csv).name} | {Path(args.checkpoint)} ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k:<15}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ── Subgroup reports ──────────────────────────────────────────────────
    subgroups = {}
    for col in SUBGROUP_COLS:
        if col not in ds.df.columns:
            continue
        values = ds.df[col].astype(str).str.strip()
        if values.replace("", np.nan).dropna().nunique() < 2:
            continue
        rep = subgroup_report(
            ds.df, o["presence_logits"], o["presence_targets"],
            o["severity_logits"], o["severity_targets"], group_col=col,
        )
        subgroups[col] = rep
        print(f"\n--- subgroups by {col} ---")
        for g, m in rep.items():
            if g == "_gaps":
                print(f"  gaps: {m}")
                continue
            line = (f"  {g:<22} n={m['n']:<4} "
                    f"sens={m.get('sensitivity', float('nan')):.3f} "
                    f"auroc={m.get('auroc', float('nan')):.3f}")
            if "qwk" in m:
                line += f" qwk={m['qwk']:.3f} within1={m['within_one']:.3f}"
            if not m["reliable"]:
                line += "  (small group — indicative only)"
            print(line)

    # ── Per-crop predictions CSV ──────────────────────────────────────────
    pred = pd.DataFrame({
        "image_path":    o["paths"],
        "presence_true": o["presence_targets"].numpy(),
        "presence_prob": torch.sigmoid(o["presence_logits"]).numpy(),
    })
    if o["severity_logits"] is not None:
        pred["severity_true"] = o["severity_targets"].numpy()
        pred["severity_pred"] = coral_logits_to_grade(o["severity_logits"]).numpy()
        pred["severity_conf"] = coral_grade_confidence(o["severity_logits"]).numpy()
        pred["abs_error"]     = (pred["severity_true"] - pred["severity_pred"]).abs()
    for col in ["subject_id", "source_dataset", "age_band", "eye"]:
        if col in ds.df.columns:
            pred[col] = ds.df[col].values

    tag = args.tag or Path(args.csv).stem
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"predictions_{tag}.csv"
    pred.to_csv(pred_path, index=False)

    payload = {
        "checkpoint": str(args.checkpoint),
        "csv":        str(args.csv),
        "n_crops":    len(ds),
        "metrics":    {k: float(v) for k, v in metrics.items()},
        "subgroups":  subgroups,
    }
    json_path = out_dir / f"metrics_{tag}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    print(f"\npredictions -> {pred_path}")
    print(f"metrics     -> {json_path}")

    # ── Severity confusion matrix (if applicable) ─────────────────────────
    if o["severity_logits"] is not None:
        conf = pd.crosstab(pred["severity_true"], pred["severity_pred"])
        conf = conf.reindex(index=range(5), columns=range(5), fill_value=0)
        print("\nSeverity confusion (rows=true, cols=pred):")
        print(conf.to_string())


if __name__ == "__main__":
    main()
