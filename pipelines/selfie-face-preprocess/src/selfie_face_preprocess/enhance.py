from __future__ import annotations

from typing import Any

import numpy as np

from .config import PreprocessConfig
from .quality import score_image


def enhance_for_skin(
    rgb: np.ndarray,
    config: PreprocessConfig,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    """Apply conservative capture correction while preserving skin texture."""

    out = rgb.astype(np.float32)
    warnings: list[str] = []
    params: dict[str, Any] = {
        "exposure_gain": 1.0,
        "white_balance_gains": [1.0, 1.0, 1.0],
        "clahe_applied": False,
        "denoise_applied": False,
    }

    if config.enable_exposure:
        out, gain = _apply_exposure(out, config, mask)
        params["exposure_gain"] = gain

    if config.enable_white_balance:
        out, gains = _apply_white_balance(out, config, mask)
        params["white_balance_gains"] = gains.tolist()

    out_u8 = np.clip(out, 0, 255).astype(np.uint8)

    if config.enable_clahe:
        out_u8, applied, warning = _apply_clahe(out_u8, config)
        params["clahe_applied"] = applied
        if warning:
            warnings.append(warning)

    if config.enable_denoise:
        current_quality = score_image(out_u8, mask)
        if current_quality["noise_estimate"] >= config.noise_threshold:
            out_u8, applied, warning = _apply_denoise(out_u8, config)
            params["denoise_applied"] = applied
            if warning:
                warnings.append(warning)

    return out_u8, params, warnings


def _apply_exposure(
    rgb: np.ndarray,
    config: PreprocessConfig,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, float]:
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    selected = _masked_values(luma, mask)
    median = float(np.median(selected)) if selected.size else float(np.median(luma))
    if median <= 1.0:
        return rgb, 1.0
    gain = config.exposure_target_luma / median
    gain = float(np.clip(gain, config.exposure_gain_min, config.exposure_gain_max))
    if abs(gain - 1.0) < 0.03:
        gain = 1.0
    return np.clip(rgb * gain, 0, 255), gain


def _apply_white_balance(
    rgb: np.ndarray,
    config: PreprocessConfig,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    pixels = rgb.reshape(-1, 3)
    if mask is not None and mask.shape == rgb.shape[:2]:
        face_pixels = rgb[np.asarray(mask) > 0]
        if face_pixels.size:
            pixels = face_pixels.reshape(-1, 3)
    means = np.maximum(pixels.mean(axis=0), 1.0)
    target = float(means.mean())
    gains = target / means
    gains = np.clip(gains, config.white_balance_gain_min, config.white_balance_gain_max)
    if np.all(np.abs(gains - 1.0) < 0.02):
        gains = np.ones(3, dtype=np.float32)
    return np.clip(rgb * gains.reshape(1, 1, 3), 0, 255), gains.astype(np.float32)


def _apply_clahe(rgb: np.ndarray, config: PreprocessConfig) -> tuple[np.ndarray, bool, str | None]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return rgb, False, "opencv_unavailable_clahe_skipped"

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=config.clahe_clip_limit, tileGridSize=(8, 8))
    corrected_l = clahe.apply(l_channel)
    blended_l = cv2.addWeighted(
        l_channel,
        1.0 - config.clahe_blend,
        corrected_l,
        config.clahe_blend,
        0,
    )
    merged = cv2.merge([blended_l, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB), True, None


def _apply_denoise(rgb: np.ndarray, config: PreprocessConfig) -> tuple[np.ndarray, bool, str | None]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return rgb, False, "opencv_unavailable_denoise_skipped"

    denoised = cv2.fastNlMeansDenoisingColored(
        rgb,
        None,
        h=config.denoise_h,
        hColor=config.denoise_h,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    blended = cv2.addWeighted(rgb, 1.0 - config.denoise_blend, denoised, config.denoise_blend, 0)
    return blended, True, None


def _masked_values(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None or mask.shape != values.shape:
        return values.reshape(-1)
    selected = values[np.asarray(mask) > 0]
    return selected if selected.size else values.reshape(-1)
