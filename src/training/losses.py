#!/usr/bin/env python3
"""
Multi-task loss for the DermaLens eye-bag model.

Combines (per the spec's loss_weights config):
    presence loss     : BCE     weight 1.0
    severity loss     : CORAL   weight 1.0
    dark circles loss : BCE     weight 0.25

The dark-circles weight is intentionally small. That head exists to force the
encoder to disentangle puffiness from discoloration — not because dark-circle
prediction is itself the product. A large weight would let it dominate training.

Usage:
    criterion = MultiTaskLoss(w_presence=1.0, w_severity=1.0, w_dark_circles=0.25)
    losses = criterion(model_output, batch)     # batch from EyeBagDataset
    losses["total"].backward()
    # For logging: losses["presence"], losses["severity"], losses["dark_circles"]
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ordinal_head import coral_loss


class MultiTaskLoss(nn.Module):
    """
    Weighted sum of the per-head losses. Heads absent from the model output
    are skipped automatically, so this same criterion works for all 3 stages
    (binary baseline, ordinal, multitask).

    Args:
        w_presence:     Weight for the presence BCE term.
        w_severity:     Weight for the CORAL severity term.
        w_dark_circles: Weight for the dark-circles BCE term.
        num_grades:     Severity grade count (for CORAL target construction).
        presence_pos_weight:
            Positive-class weight for presence BCE. If you have many more
            negatives than positives even after balanced sampling, set this
            to roughly (n_negative / n_positive). None = no reweighting.
    """

    def __init__(
        self,
        w_presence:     float = 1.0,
        w_severity:     float = 1.0,
        w_dark_circles: float = 0.25,
        num_grades:     int   = 5,
        presence_pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.w_presence     = w_presence
        self.w_severity     = w_severity
        self.w_dark_circles = w_dark_circles
        self.num_grades     = num_grades

        if presence_pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([presence_pos_weight]))
        else:
            self.pos_weight = None

    def forward(
        self,
        output: Dict[str, torch.Tensor],   # from EyeBagModel.forward()
        batch:  Dict,                      # from DataLoader (EyeBagDataset samples)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict of scalar tensors:
            "total"        — weighted sum, call .backward() on this
            "presence"     — detached, for logging
            "severity"     — detached, for logging (0.0 if head disabled)
            "dark_circles" — detached, for logging (0.0 if head disabled)
        """
        device = output["presence_logit"].device
        zero   = torch.tensor(0.0, device=device)

        # ── Presence ──────────────────────────────────────────────────────
        loss_presence = F.binary_cross_entropy_with_logits(
            output["presence_logit"],
            batch["presence"].to(device),
            pos_weight=self.pos_weight,
        )

        # ── Severity (CORAL) ──────────────────────────────────────────────
        if "severity_logits" in output:
            loss_severity = coral_loss(
                output["severity_logits"],
                batch["severity"].to(device),
                self.num_grades,
            )
        else:
            loss_severity = zero

        # ── Dark circles ──────────────────────────────────────────────────
        if "dark_circles_logit" in output:
            loss_dc = F.binary_cross_entropy_with_logits(
                output["dark_circles_logit"],
                batch["dark_circles"].to(device),
            )
        else:
            loss_dc = zero

        total = (
            self.w_presence     * loss_presence +
            self.w_severity     * loss_severity +
            self.w_dark_circles * loss_dc
        )

        return {
            "total":        total,
            "presence":     loss_presence.detach(),
            "severity":     loss_severity.detach() if torch.is_tensor(loss_severity) else zero,
            "dark_circles": loss_dc.detach()       if torch.is_tensor(loss_dc)       else zero,
        }
