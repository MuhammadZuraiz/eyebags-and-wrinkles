#!/usr/bin/env python3
"""
Main training script for the DermaLens eye-bag model.

Runs any of the 3 sprint stages from a YAML config:

    Day 5 — binary baseline:
        python scripts/train.py --config configs/baseline_binary.yaml

    Day 7 — ordinal severity:
        python scripts/train.py --config configs/ordinal_severity.yaml

    Day 8 — full multi-task:
        python scripts/train.py --config configs/multitask.yaml

    Warm-start a later stage from an earlier checkpoint:
        python scripts/train.py --config configs/ordinal_severity.yaml \
            --warm-start experiments/baseline_binary/best.pt

    Quick smoke test (tiny encoder, 2 epochs, no pretrained download):
        python scripts/train.py --config configs/baseline_binary.yaml --smoke

On Google Colab:
    1. Upload the repo (or git clone it).
    2. Runtime → Change runtime type → T4 GPU.
    3. !pip install -r requirements.txt
    4. !python scripts/train.py --config configs/baseline_binary.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset       import EyeBagDataset
from src.data.sampler       import build_balanced_sampler
from src.data.augmentations import get_train_transforms, get_val_transforms
from src.models.multitask   import EyeBagModel
from src.training.losses    import MultiTaskLoss
from src.training.trainer   import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train the DermaLens eye-bag model")
    parser.add_argument("--config",     required=True, help="Path to YAML config")
    parser.add_argument("--warm-start", default="",
                        help="Checkpoint to load encoder weights from (previous stage)")
    parser.add_argument("--resume",     default="",
                        help="Path to last.pt to continue an interrupted run "
                             "(restores optimizer/scheduler/early-stopping state)")
    parser.add_argument("--smoke",      action="store_true",
                        help="Smoke test: resnet18, no pretrained weights, 2 epochs")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    # Explicit utf-8: configs contain non-ASCII comments and Windows would
    # otherwise read them as cp1252 and crash.
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg  = cfg.get("data", {})
    loss_cfg  = cfg.get("loss_weights", {})

    if args.smoke:
        logger.info("SMOKE TEST MODE: small encoder, 2 epochs, no pretrained download")
        model_cfg["encoder"]    = "resnet18"
        model_cfg["pretrained"] = False
        train_cfg["epochs"]     = 2
        train_cfg["mixed_precision"] = False

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = EyeBagDataset(
        data_cfg["train_csv"],
        transform=get_train_transforms(),
        crops_dir=data_cfg.get("crops_dir"),
    )
    val_ds = EyeBagDataset(
        data_cfg["val_csv"],
        transform=get_val_transforms(),
        crops_dir=data_cfg.get("crops_dir"),
    )

    # ── Loaders ───────────────────────────────────────────────────────────
    batch_size  = train_cfg.get("batch_size", 64)
    num_workers = train_cfg.get("num_workers", 2)
    pin = torch.cuda.is_available()

    sampler = build_balanced_sampler(
        train_ds,
        balance_by=data_cfg.get("balance_by", "severity"),
        smoothing=data_cfg.get("sampler_smoothing", 0.5),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    # Resolve every architecture choice up front; this exact dict travels
    # inside checkpoints so inference/export can rebuild the model.
    resolved_model_cfg = {
        "encoder":               model_cfg.get("encoder", "convnext_tiny"),
        "pretrained":            model_cfg.get("pretrained", True),
        "severity_grades":       model_cfg.get("severity_grades", 5),
        "proj_dim":              model_cfg.get("proj_dim", 512),
        "dropout":               model_cfg.get("dropout", 0.2),
        "use_severity_head":     model_cfg.get("use_severity_head", True),
        "use_dark_circles_head": model_cfg.get("use_dark_circles_head", True),
    }
    model = EyeBagModel(
        encoder_name     = resolved_model_cfg["encoder"],
        pretrained       = resolved_model_cfg["pretrained"],
        num_grades       = resolved_model_cfg["severity_grades"],
        proj_dim         = resolved_model_cfg["proj_dim"],
        dropout          = resolved_model_cfg["dropout"],
        use_severity     = resolved_model_cfg["use_severity_head"],
        use_dark_circles = resolved_model_cfg["use_dark_circles_head"],
    )

    if args.warm_start:
        model.load_encoder_from(args.warm_start)

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = MultiTaskLoss(
        w_presence     = loss_cfg.get("presence_bce", 1.0),
        w_severity     = loss_cfg.get("severity_corn", 1.0),
        w_dark_circles = loss_cfg.get("confounders_bce", 0.25),
        num_grades     = model_cfg.get("severity_grades", 5),
        presence_pos_weight = loss_cfg.get("presence_pos_weight"),
    )

    # ── Train ─────────────────────────────────────────────────────────────
    out_dir = cfg.get("output_dir", f"experiments/{Path(args.config).stem}")
    trainer = Trainer(
        model        = model,
        criterion    = criterion,
        train_loader = train_loader,
        val_loader   = val_loader,
        config       = train_cfg,
        out_dir      = out_dir,
        model_config = resolved_model_cfg,
    )
    trainer.fit(resume_from=args.resume or None)

    # ── Final summary ─────────────────────────────────────────────────────
    final_metrics = trainer.evaluate(val_loader)
    logger.info("Final validation metrics:")
    for k, v in sorted(final_metrics.items()):
        logger.info(f"  {k:<15}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nNext step:")
    print(f"  python scripts/error_analysis.py --checkpoint {out_dir}/best.pt "
          f"--csv {data_cfg['val_csv']}")


if __name__ == "__main__":
    main()
