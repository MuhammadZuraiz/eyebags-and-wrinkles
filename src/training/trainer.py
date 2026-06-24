#!/usr/bin/env python3
"""
Training loop for the DermaLens eye-bag model.

Features (matching the spec's training config):
    - AdamW with separate encoder/head learning rates
    - Linear warmup → cosine decay schedule
    - Automatic mixed precision (AMP) when a CUDA GPU is available
    - Early stopping on validation metric
    - Best-checkpoint saving (+ a `last.pt` for resuming)
    - Per-epoch metrics logging to console and history.json

You normally don't call this directly — use scripts/train.py which wires up
the config, dataset, sampler and model. But the Trainer class is importable
for notebooks (e.g. Colab).

Usage (programmatic):
    trainer = Trainer(model, criterion, train_loader, val_loader, config, out_dir)
    history = trainer.fit()
"""

import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Args:
        model:        EyeBagModel instance.
        criterion:    MultiTaskLoss instance.
        train_loader: DataLoader over the training split (with balanced sampler).
        val_loader:   DataLoader over the validation split (NO augmentation).
        config:       Dict with keys (all have sensible defaults):
                        epochs, encoder_lr, head_lr, weight_decay,
                        warmup_epochs, early_stopping_patience,
                        monitor ("val_qwk" or "val_auroc" or "val_loss"),
                        mixed_precision (bool)
        out_dir:      Directory for checkpoints + history.json.
        model_config: Resolved architecture dict (encoder, severity_grades,
                      proj_dim, dropout, use_severity_head, use_dark_circles_head).
                      Saved into every checkpoint so inference/export can rebuild
                      the exact model without guessing from state-dict keys.
    """

    DEFAULTS = {
        "epochs":                  80,
        "encoder_lr":              3e-5,
        "head_lr":                 3e-4,
        "weight_decay":            1e-4,
        "warmup_epochs":           3,
        "early_stopping_patience": 12,
        "monitor":                 "val_auroc",   # binary stage; switch to val_qwk for ordinal
        "mixed_precision":         True,
        "grad_clip_norm":          1.0,
    }

    def __init__(
        self,
        model:        nn.Module,
        criterion:    nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        config:       Optional[Dict] = None,
        out_dir:      str = "experiments/run",
        model_config: Optional[Dict] = None,
    ):
        self.cfg = {**self.DEFAULTS, **(config or {})}
        self.model_config = dict(model_config or {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model        = model.to(self.device)
        self.criterion    = criterion.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # ── Optimiser: two LR groups (pretrained encoder vs fresh heads) ──
        if hasattr(model, "parameter_groups"):
            param_groups = model.parameter_groups(
                encoder_lr=self.cfg["encoder_lr"],
                head_lr=self.cfg["head_lr"],
            )
        else:
            param_groups = [{"params": model.parameters(), "lr": self.cfg["head_lr"]}]

        self.optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.cfg["weight_decay"]
        )

        # ── LR schedule: linear warmup then cosine decay ──────────────────
        total_steps  = self.cfg["epochs"] * max(1, len(train_loader))
        warmup_steps = self.cfg["warmup_epochs"] * max(1, len(train_loader))

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # ── AMP (mixed precision) ──────────────────────────────────────────
        self.use_amp = self.cfg["mixed_precision"] and self.device.type == "cuda"
        self.scaler  = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Best-metric tracking lives on the instance so checkpoints can carry
        # it and a resumed run continues early stopping where it left off.
        monitor = self.cfg["monitor"]
        self._higher_is_better = not monitor.endswith("loss")
        self._best_value    = -np.inf if self._higher_is_better else np.inf
        self._best_epoch    = -1
        self._patience_left = self.cfg["early_stopping_patience"]
        self._history       = {"epochs": [], "train_loss": [], "val_metrics": [], "lr": []}

        logger.info(
            f"Trainer ready: device={self.device}  amp={self.use_amp}  "
            f"epochs={self.cfg['epochs']}  monitor={self.cfg['monitor']}"
        )

    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, resume_from: Optional[str] = None) -> Dict:
        """
        Run the full training loop.

        Args:
            resume_from: Path to a previous last.pt to continue an interrupted
                         run (restores model, optimizer, scheduler, scaler,
                         best-metric tracking and history).

        Returns the history dict:
            {"epochs": [...], "train_loss": [...], "val_metrics": [...], "best_epoch": int}
        """
        monitor = self.cfg["monitor"]
        start_epoch = 1
        if resume_from:
            start_epoch = self._load_resume_state(resume_from) + 1
            logger.info(f"Resumed from {resume_from}: continuing at epoch {start_epoch}")

        for epoch in range(start_epoch, self.cfg["epochs"] + 1):
            t0 = time.time()

            train_loss = self._train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)

            current = val_metrics.get(monitor.replace("val_", ""), None)
            if current is None:
                # Fall back to val loss if the chosen monitor isn't computed
                current = val_metrics.get("loss", 0.0)
                self._higher_is_better = False

            lr_now = self.optimizer.param_groups[0]["lr"]
            self._history["epochs"].append(epoch)
            self._history["train_loss"].append(train_loss)
            self._history["val_metrics"].append(val_metrics)
            self._history["lr"].append(lr_now)

            improved = (
                (current > self._best_value) if self._higher_is_better
                else (current < self._best_value)
            )
            marker = ""
            if improved:
                self._best_value    = current
                self._best_epoch    = epoch
                self._patience_left = self.cfg["early_stopping_patience"]
                self._save_checkpoint("best.pt", epoch, val_metrics)
                marker = "  ★ best"
            else:
                self._patience_left -= 1

            self._save_checkpoint("last.pt", epoch, val_metrics)

            dt = time.time() - t0
            logger.info(
                f"Epoch {epoch:>3}/{self.cfg['epochs']}  "
                f"train_loss={train_loss:.4f}  "
                f"{monitor}={current:.4f}  "
                f"lr={lr_now:.2e}  "
                f"({dt:.0f}s){marker}"
            )

            if self._patience_left <= 0:
                logger.info(
                    f"Early stopping: no {monitor} improvement for "
                    f"{self.cfg['early_stopping_patience']} epochs. "
                    f"Best was {self._best_value:.4f} at epoch {self._best_epoch}."
                )
                break

        history = dict(self._history)
        history["best_epoch"] = self._best_epoch
        history["best_value"] = float(self._best_value)
        (self.out_dir / "history.json").write_text(json.dumps(history, indent=2, default=float))
        logger.info(f"Training complete. Best {monitor}={self._best_value:.4f} @ epoch {self._best_epoch}")
        logger.info(f"Checkpoints in {self.out_dir}/  (best.pt, last.pt)")
        return history

    def _load_resume_state(self, path: str) -> int:
        """Restore full training state from a checkpoint. Returns its epoch."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state"):
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if ckpt.get("scaler_state") and self.use_amp:
            self.scaler.load_state_dict(ckpt["scaler_state"])
        if "best_value" in ckpt:
            self._best_value = ckpt["best_value"]
        if "best_epoch" in ckpt:
            self._best_epoch = ckpt["best_epoch"]
        if "patience_left" in ckpt:
            self._patience_left = ckpt["patience_left"]
        if ckpt.get("history"):
            self._history = ckpt["history"]
        return int(ckpt["epoch"])

    # ─────────────────────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, n_batches = 0.0, 0

        for batch in self.train_loader:
            images = batch["image"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                output = self.model(images)
                losses = self.criterion(output, batch)

            self.scaler.scale(losses["total"]).backward()

            # Gradient clipping (must unscale first under AMP)
            if self.cfg["grad_clip_norm"]:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg["grad_clip_norm"]
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += losses["total"].item()
            n_batches  += 1

        return total_loss / max(1, n_batches)

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict:
        """
        Run the model over a loader and compute all metrics.
        Returns a flat dict of floats (see src/evaluation/metrics.py).
        """
        self.model.eval()

        all_presence_logits, all_presence_targets = [], []
        all_severity_logits, all_severity_targets = [], []
        all_dc_logits,       all_dc_targets       = [], []
        total_loss, n_batches = 0.0, 0

        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                output = self.model(images)
                losses = self.criterion(output, batch)

            total_loss += losses["total"].item()
            n_batches  += 1

            all_presence_logits.append(output["presence_logit"].float().cpu())
            all_presence_targets.append(batch["presence"])

            if "severity_logits" in output:
                all_severity_logits.append(output["severity_logits"].float().cpu())
                all_severity_targets.append(batch["severity"])

            if "dark_circles_logit" in output:
                all_dc_logits.append(output["dark_circles_logit"].float().cpu())
                all_dc_targets.append(batch["dark_circles"])

        metrics = compute_metrics(
            presence_logits  = torch.cat(all_presence_logits),
            presence_targets = torch.cat(all_presence_targets),
            severity_logits  = torch.cat(all_severity_logits)  if all_severity_logits else None,
            severity_targets = torch.cat(all_severity_targets) if all_severity_targets else None,
            dc_logits        = torch.cat(all_dc_logits)        if all_dc_logits else None,
            dc_targets       = torch.cat(all_dc_targets)       if all_dc_targets else None,
        )
        metrics["loss"] = total_loss / max(1, n_batches)
        return metrics

    # ─────────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, name: str, epoch: int, val_metrics: Dict):
        torch.save(
            {
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state":    self.scaler.state_dict(),
                "epoch":           epoch,
                "val_metrics":     val_metrics,
                "config":          self.cfg,
                "model_config":    self.model_config,
                "best_value":      float(self._best_value),
                "best_epoch":      self._best_epoch,
                "patience_left":   self._patience_left,
                "history":         self._history,
            },
            self.out_dir / name,
        )
