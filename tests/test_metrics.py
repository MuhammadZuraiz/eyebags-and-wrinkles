"""Tests for the metrics module."""
import numpy as np
import torch

from src.evaluation.metrics import (
    binary_metrics, quadratic_weighted_kappa, severity_metrics,
    expected_calibration_error,
)


def test_perfect_qwk():
    y = np.array([0, 1, 2, 3, 4, 2, 1])
    assert quadratic_weighted_kappa(y, y) == 1.0


def test_qwk_penalises_distance():
    y_true = np.array([0, 0, 4, 4] * 10)
    off_by_one  = np.array([1, 1, 3, 3] * 10)
    off_by_four = np.array([4, 4, 0, 0] * 10)
    assert quadratic_weighted_kappa(y_true, off_by_one) > \
           quadratic_weighted_kappa(y_true, off_by_four)


def test_binary_metrics_keys():
    logits  = torch.randn(100)
    targets = (torch.rand(100) > 0.5).float()
    m = binary_metrics(logits, targets)
    for k in ["sensitivity", "specificity", "precision", "f1", "auroc", "ece"]:
        assert k in m


def test_within_one_grade():
    from src.models.ordinal_head import CoralHead
    torch.manual_seed(0)
    # decisive grade-2 logits vs grade-2/3 truth
    logits = torch.tensor([[5., 5., -5., -5.]] * 10)
    truths = torch.tensor([2] * 5 + [3] * 5)
    m = severity_metrics(logits, truths)
    assert m["within_one"] == 1.0
    assert m["exact"] == 0.5


def test_ece_perfect_calibration():
    probs = np.array([0.1] * 100 + [0.9] * 100)
    y     = np.array([0]*90 + [1]*10 + [1]*90 + [0]*10)
    assert expected_calibration_error(probs, y) < 0.05
