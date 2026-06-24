from __future__ import annotations

from typing import Any

import numpy as np

from .config import PreprocessConfig


def luminance(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    return 0.2126 * rgb_f[..., 0] + 0.7152 * rgb_f[..., 1] + 0.0722 * rgb_f[..., 2]


def score_image(rgb: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    """Compute capture-quality metrics without altering the image."""

    luma = luminance(rgb)
    selected = _select_masked(luma, mask)
    if selected.size == 0:
        selected = luma.reshape(-1)

    glare = (selected >= 245).mean() if selected.size else 0.0
    return {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "luminance_mean": float(np.mean(selected)),
        "luminance_median": float(np.median(selected)),
        "luminance_std": float(np.std(selected)),
        "blur_laplacian_var": float(laplacian_variance(luma)),
        "glare_fraction": float(glare),
        "noise_estimate": float(estimate_noise(luma)),
    }


def quality_reject_reasons(quality: dict[str, Any], config: PreprocessConfig) -> list[str]:
    reasons: list[str] = []
    if quality["blur_laplacian_var"] < config.blur_threshold:
        reasons.append("too_blurry")
    if quality["luminance_median"] < config.min_luminance:
        reasons.append("too_dark")
    if quality["luminance_median"] > config.max_luminance:
        reasons.append("too_bright")
    if quality["glare_fraction"] > config.max_glare_fraction:
        reasons.append("heavy_glare")
    return reasons


def laplacian_variance(gray: np.ndarray) -> float:
    try:
        import cv2  # type: ignore
    except ImportError:
        center = gray[1:-1, 1:-1]
        lap = (
            gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
            - 4.0 * center
        )
        return float(np.var(lap)) if lap.size else 0.0

    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return float(lap.var())


def estimate_noise(gray: np.ndarray) -> float:
    try:
        import cv2  # type: ignore
    except ImportError:
        padded = np.pad(gray, 1, mode="edge")
        box = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0
        residual = gray - box
        return float(np.std(residual))

    blurred = cv2.blur(gray.astype(np.float32), (3, 3))
    return float(np.std(gray.astype(np.float32) - blurred))


def _select_masked(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return values.reshape(-1)
    mask_bool = np.asarray(mask) > 0
    if mask_bool.shape != values.shape:
        return values.reshape(-1)
    return values[mask_bool]
