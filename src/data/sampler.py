#!/usr/bin/env python3
"""
Class-balanced sampling for DermaLens training.

THE PROBLEM:
  Real-world skin data is heavily imbalanced. You'll likely have:
    - Many grade-0 (no eye bags) crops
    - A decent number of grade-1/2
    - Very few grade-3/4 (pronounced cases are rarer in casual datasets)

  A plain shuffled DataLoader feeds the model mostly grade-0 examples.
  The model then learns the lazy strategy — "predict 0 every time" — and
  scores high accuracy while being useless.

THE FIX:
  WeightedRandomSampler gives each sample a probability inversely related to
  how common its class is, so rare grades are seen more often per epoch.

Usage:
    from torch.utils.data import DataLoader
    from src.data.dataset import EyeBagDataset
    from src.data.sampler import build_balanced_sampler

    train_ds = EyeBagDataset("data/splits/train.csv", transform=train_tf)
    sampler  = build_balanced_sampler(train_ds)

    # IMPORTANT: shuffle must be False (and not passed) when using a sampler
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=4)
"""

import logging
from collections import Counter
from typing import Optional

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

logger = logging.getLogger(__name__)


def build_balanced_sampler(
    dataset,                           # EyeBagDataset
    balance_by:  str   = "severity",   # "severity" or "presence"
    smoothing:   float = 0.5,          # 0 = full inverse-frequency, 1 = no rebalancing
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that oversamples rare severity grades.

    Args:
        dataset:    EyeBagDataset (must have .df with a "severity" column).
        balance_by: "severity" → balance across the 5 grades (recommended)
                    "presence" → balance positive vs negative only
        smoothing:  How aggressively to rebalance:
                      weight = 1 / count^(1 - smoothing)
                    - 0.0 → strict inverse frequency (rare grades sampled a LOT)
                    - 0.5 → square-root inverse frequency (gentler — recommended)
                    - 1.0 → no rebalancing
                    Start at 0.5. If the model still ignores rare grades, lower it.
        num_samples: Samples per epoch. Default = len(dataset).

    Returns:
        WeightedRandomSampler for DataLoader(sampler=...).
    """
    df = dataset.df

    if balance_by == "severity":
        keys = df["severity"].to_numpy()
    elif balance_by == "presence":
        keys = (df["severity"].to_numpy() > 0).astype(int)
    else:
        raise ValueError(f"balance_by must be 'severity' or 'presence', got {balance_by!r}")

    counts = Counter(keys.tolist())
    logger.info(f"Sampler class counts ({balance_by}): {dict(sorted(counts.items()))}")

    exponent = 1.0 - float(np.clip(smoothing, 0.0, 1.0))
    class_weight = {cls: 1.0 / (cnt ** exponent) for cls, cnt in counts.items()}

    sample_weights = np.array([class_weight[k] for k in keys], dtype=np.float64)
    sample_weights /= sample_weights.sum()

    for cls in sorted(counts):
        eff = sample_weights[keys == cls].sum() * 100
        logger.info(
            f"  class {cls}: {counts[cls]:>5} samples → "
            f"{eff:.1f}% of each epoch (raw share: {counts[cls]/len(keys)*100:.1f}%)"
        )

    return WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights),
        num_samples = num_samples or len(dataset),
        replacement = True,
    )
