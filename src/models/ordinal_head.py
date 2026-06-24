#!/usr/bin/env python3
"""
CORAL ordinal regression head for severity grading.

WHY ORDINAL REGRESSION (not plain classification):
  Severity grades are ORDERED: 0 < 1 < 2 < 3 < 4.
  A normal 5-class softmax classifier treats "predicted 3, truth 2" exactly the
  same as "predicted 0, truth 4". That's wrong — being off by one grade is a
  minor error; being off by four grades is a catastrophic one.

HOW CORAL WORKS (the intuition):
  Instead of asking "which of the 5 grades is it?", CORAL asks 4 binary questions:
      Q1: Is the severity > 0 ?
      Q2: Is the severity > 1 ?
      Q3: Is the severity > 2 ?
      Q4: Is the severity > 3 ?
  The predicted grade = number of "yes" answers.
  Example: yes, yes, no, no → grade 2.

  The "rank-consistency" trick: all 4 questions share the SAME weight vector
  and differ only by a learned bias term. This guarantees the answers are
  monotonic (you can never get "no, yes" — i.e. P(>1) can't exceed P(>0)),
  which a naive multi-binary-head setup does not guarantee.

  Reference: Cao, Mirjalili, Raschka (2019) — "Rank consistent ordinal
  regression for neural networks" (arXiv:1901.07884).

Usage:
    head = CoralHead(in_features=512, num_grades=5)
    logits = head(features)                 # (B, 4) — one logit per threshold
    probas = torch.sigmoid(logits)          # P(grade > k) for k = 0..3
    grades = coral_logits_to_grade(logits)  # (B,) int — predicted grade 0–4

    loss = coral_loss(logits, targets)      # targets: (B,) int64 grades 0–4
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralHead(nn.Module):
    """
    CORAL ordinal regression output head.

    Architecture:
        One shared Linear layer with a SINGLE output unit (the shared weight
        vector w), plus (num_grades - 1) bias terms.

        logit_k = w·x + b_k     for k = 0 .. num_grades-2

    Because all thresholds share w, rank consistency holds exactly when the
    biases are descending. Independent bias parameters are only descending at
    initialisation — gradient updates can reorder them. So the biases are
    parameterised as cumulative softplus offsets from a top bias:

        b_0 = top
        b_k = top - sum_{i<k} softplus(delta_i)      (strictly descending)

    which makes rank consistency a structural guarantee, not an initial
    condition.

    Args:
        in_features: Dimension of the input feature vector.
        num_grades:  Number of ordinal classes (5 for grades 0–4).
    """

    def __init__(self, in_features: int, num_grades: int = 5):
        super().__init__()
        self.num_grades     = num_grades
        self.num_thresholds = num_grades - 1   # 4 binary questions for 5 grades

        # Shared weight vector — bias=False because the biases come separately
        self.fc = nn.Linear(in_features, 1, bias=False)

        # Initialise to reproduce linspace(2.0, -2.0, num_thresholds): top bias
        # 2.0, equal gaps of 4/(T-1) encoded through softplus's inverse.
        self.bias_top = nn.Parameter(torch.tensor(2.0))
        if self.num_thresholds > 1:
            gap = 4.0 / (self.num_thresholds - 1)
            inv_softplus_gap = math.log(math.expm1(gap))
            self.bias_deltas = nn.Parameter(
                torch.full((self.num_thresholds - 1,), inv_softplus_gap)
            )
        else:
            self.bias_deltas = nn.Parameter(torch.empty(0))

    @property
    def biases(self) -> torch.Tensor:
        """Threshold biases, strictly descending by construction."""
        offsets = torch.cat([
            torch.zeros(1, device=self.bias_top.device, dtype=self.bias_top.dtype),
            torch.cumsum(F.softplus(self.bias_deltas), dim=0),
        ])
        return self.bias_top - offsets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: feature tensor, shape (B, in_features)
        Returns:
            logits, shape (B, num_thresholds) — logit_k = P(grade > k) pre-sigmoid
        """
        shared = self.fc(x)                  # (B, 1)
        return shared + self.biases          # broadcast → (B, num_thresholds)


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def coral_targets(grades: torch.Tensor, num_grades: int = 5) -> torch.Tensor:
    """
    Convert integer grades to CORAL binary target matrix.

    Example for num_grades=5:
        grade 0 → [0, 0, 0, 0]
        grade 2 → [1, 1, 0, 0]
        grade 4 → [1, 1, 1, 1]

    Args:
        grades: (B,) int64 tensor of grades 0..num_grades-1
    Returns:
        (B, num_grades-1) float tensor of binary targets
    """
    num_thresholds = num_grades - 1
    # levels[k] = 1 if grade > k
    thresholds = torch.arange(num_thresholds, device=grades.device)   # [0,1,2,3]
    return (grades.unsqueeze(1) > thresholds.unsqueeze(0)).float()


def coral_loss(
    logits:  torch.Tensor,   # (B, num_thresholds)
    grades:  torch.Tensor,   # (B,) int64
    num_grades: int = 5,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    CORAL loss = sum of binary cross-entropies over the threshold questions.

    Each threshold question is a binary classification, so the total loss is
    just BCE applied to all of them. The ordering knowledge comes from the
    TARGET structure (cumulative 1s), not from a special loss formula.
    """
    targets = coral_targets(grades, num_grades)             # (B, T)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction=reduction)


# ──────────────────────────────────────────────────────────────────────────────
# Decoding
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def coral_logits_to_grade(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CORAL logits to predicted integer grades.

    predicted_grade = number of thresholds where P(grade > k) > 0.5

    Args:
        logits: (B, num_thresholds)
    Returns:
        (B,) int64 tensor of grades 0..num_thresholds
    """
    probas = torch.sigmoid(logits)
    return (probas > 0.5).sum(dim=1).long()


@torch.no_grad()
def coral_grade_confidence(logits: torch.Tensor) -> torch.Tensor:
    """
    A simple confidence score for the predicted grade.

    Intuition: confidence is high when every threshold probability is far from
    0.5 (the model is decisively answering yes or no on every question), and
    low when any threshold probability hovers near 0.5.

        confidence = mean over thresholds of  2 * |P_k - 0.5|

    Args:
        logits: (B, num_thresholds)
    Returns:
        (B,) float tensor in [0, 1]
    """
    probas = torch.sigmoid(logits)
    return (2.0 * (probas - 0.5).abs()).mean(dim=1)
