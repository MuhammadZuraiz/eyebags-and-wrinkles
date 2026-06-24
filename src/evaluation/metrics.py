#!/usr/bin/env python3
"""
Evaluation metrics for the DermaLens eye-bag model.

Implements the spec's metric requirements:
    Presence : sensitivity, specificity, precision, F1, AUROC, AUPRC
    Severity : MAE, exact-grade accuracy, within-one-grade accuracy,
               quadratic-weighted kappa (QWK)
    Calibration : expected calibration error (ECE), Brier score
    Subgroups : any metric computed per MST shade / lighting / device group

Quadratic-weighted kappa (QWK) — the headline severity metric:
    Measures agreement between predicted and true grades, weighted so that
    bigger grade errors are penalised quadratically. 1.0 = perfect, 0 = chance.
    Spec target: ≥ 0.75 for release.

Usage:
    from src.evaluation.metrics import compute_metrics
    metrics = compute_metrics(presence_logits, presence_targets,
                              severity_logits, severity_targets)
    print(metrics["auroc"], metrics["qwk"], metrics["within_one"])
"""

import logging
from typing import Dict, Optional

import numpy as np
import torch

from src.models.ordinal_head import coral_logits_to_grade

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Binary metrics (presence / dark circles)
# ──────────────────────────────────────────────────────────────────────────────

def binary_metrics(logits: torch.Tensor, targets: torch.Tensor, prefix: str = "") -> Dict:
    """
    Compute binary classification metrics from raw logits.

    Args:
        logits:  (N,) raw model outputs (pre-sigmoid)
        targets: (N,) float 0/1 ground truth
        prefix:  key prefix, e.g. "dc_" for the dark-circles head
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score, precision_score,
        recall_score, brier_score_loss,
    )

    probs = torch.sigmoid(logits).numpy()
    y     = targets.numpy().astype(int)
    pred  = (probs > 0.5).astype(int)

    out = {}

    # Confusion components
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    out[f"{prefix}sensitivity"] = tp / max(tp + fn, 1)   # = recall on positives
    out[f"{prefix}specificity"] = tn / max(tn + fp, 1)
    out[f"{prefix}precision"]   = tp / max(tp + fp, 1)
    out[f"{prefix}f1"]          = float(f1_score(y, pred, zero_division=0))
    out[f"{prefix}accuracy"]    = (tp + tn) / max(len(y), 1)

    # Threshold-free metrics need both classes present
    if len(np.unique(y)) == 2:
        out[f"{prefix}auroc"] = float(roc_auc_score(y, probs))
        out[f"{prefix}auprc"] = float(average_precision_score(y, probs))
        out[f"{prefix}brier"] = float(brier_score_loss(y, probs))
        out[f"{prefix}ece"]   = expected_calibration_error(probs, y)
    else:
        out[f"{prefix}auroc"] = float("nan")
        out[f"{prefix}auprc"] = float("nan")

    return out


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE: average |confidence − accuracy| across probability bins.
    0 = perfectly calibrated. Spec requires "no obvious calibration failure".
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs > lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc  = y[mask].mean()
        ece += mask.mean() * abs(bin_conf - bin_acc)
    return float(ece)


# ──────────────────────────────────────────────────────────────────────────────
# Ordinal severity metrics
# ──────────────────────────────────────────────────────────────────────────────

def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_grades: int = 5) -> float:
    """
    QWK from scratch (no sklearn dependency quirks).

    kappa = 1 − (sum(W·O) / sum(W·E))
    where O = observed confusion matrix, E = expected under independence,
    W[i,j] = (i−j)² / (G−1)²  — quadratic distance penalty.
    """
    G = num_grades
    O = np.zeros((G, G), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        O[int(t), int(p)] += 1

    if O.sum() == 0:
        return float("nan")

    W = np.zeros((G, G))
    for i in range(G):
        for j in range(G):
            W[i, j] = ((i - j) ** 2) / ((G - 1) ** 2)

    row = O.sum(axis=1)
    col = O.sum(axis=0)
    E   = np.outer(row, col) / O.sum()

    denom = (W * E).sum()
    if denom == 0:
        return float("nan")
    return float(1.0 - (W * O).sum() / denom)


def severity_metrics(
    severity_logits:  torch.Tensor,   # (N, num_grades-1) CORAL logits
    severity_targets: torch.Tensor,   # (N,) int grades
    num_grades: int = 5,
    prefix: str = "",
) -> Dict:
    pred = coral_logits_to_grade(severity_logits).numpy()
    true = severity_targets.numpy()

    abs_err = np.abs(pred - true)
    return {
        f"{prefix}mae":        float(abs_err.mean()),
        f"{prefix}exact":      float((abs_err == 0).mean()),
        f"{prefix}within_one": float((abs_err <= 1).mean()),   # spec target ≥ 0.90
        f"{prefix}qwk":        quadratic_weighted_kappa(true, pred, num_grades),  # spec ≥ 0.75
    }


# ──────────────────────────────────────────────────────────────────────────────
# Combined entry point (used by Trainer.evaluate)
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    presence_logits:  torch.Tensor,
    presence_targets: torch.Tensor,
    severity_logits:  Optional[torch.Tensor] = None,
    severity_targets: Optional[torch.Tensor] = None,
    dc_logits:        Optional[torch.Tensor] = None,
    dc_targets:       Optional[torch.Tensor] = None,
) -> Dict:
    """Compute every available metric given whichever heads are active."""
    metrics = binary_metrics(presence_logits, presence_targets)

    if severity_logits is not None and severity_targets is not None:
        metrics.update(severity_metrics(severity_logits, severity_targets))

    if dc_logits is not None and dc_targets is not None:
        metrics.update(binary_metrics(dc_logits, dc_targets, prefix="dc_"))

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Subgroup (fairness) reporting
# ──────────────────────────────────────────────────────────────────────────────

def subgroup_report(
    df_meta,                          # pandas DataFrame aligned with predictions
    presence_logits:  torch.Tensor,
    presence_targets: torch.Tensor,
    severity_logits:  Optional[torch.Tensor],
    severity_targets: Optional[torch.Tensor],
    group_col: str = "mst_shade",
    min_group_size: int = 30,
) -> Dict:
    """
    Compute metrics PER SUBGROUP (e.g. per MST shade, per lighting category).

    The spec's release gates require:
      - worst sufficiently sized MST-group sensitivity ≥ 85%
      - sensitivity gap between major MST groups ≤ 5 points

    Groups with fewer than min_group_size samples are reported but flagged
    as too small to trust.

    Returns:
        {group_value: {metrics...}, "_gaps": {"sensitivity_gap": float, ...}}
    """
    report = {}
    groups = df_meta[group_col].fillna("unknown")

    for g in sorted(groups.unique(), key=str):
        mask_np = (groups == g).to_numpy()
        mask    = torch.from_numpy(mask_np)
        n = int(mask.sum())
        if n == 0:
            continue

        entry = {"n": n, "reliable": n >= min_group_size}
        entry.update(binary_metrics(presence_logits[mask], presence_targets[mask]))
        if severity_logits is not None:
            entry.update(severity_metrics(severity_logits[mask], severity_targets[mask]))
        report[str(g)] = entry

    # Gap analysis across reliable groups only
    reliable = {g: m for g, m in report.items() if m.get("reliable")}
    if len(reliable) >= 2:
        sens = [m["sensitivity"] for m in reliable.values()]
        report["_gaps"] = {
            "sensitivity_gap":   float(max(sens) - min(sens)),      # spec ≤ 0.05
            "worst_sensitivity": float(min(sens)),                  # spec ≥ 0.85
            "groups_compared":   list(reliable.keys()),
        }
    return report
