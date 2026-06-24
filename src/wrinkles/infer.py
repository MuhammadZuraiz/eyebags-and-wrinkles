#!/usr/bin/env python3
"""
On-device wrinkle inference (onnxruntime, torch-free).

Pipeline for one selfie:
    rgb [+ landmarks]
        -> prepare_wrinkle_input  (1024 masked face + texture map)
        -> WrinkleAnalyzer.run     (4-ch input -> U-Net ONNX -> argmax mask)
        -> coverage + per-region scores (via landmark polygons)

The U-Net's input normalisation matches the labhai reference `inference.py`:
RGB and the texture map are each scaled to [-1, 1], then concatenated to a
(1, 4, 1024, 1024) tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .preprocess import OUTPUT_SIZE, prepare_wrinkle_input
from .regions import REGION_NAMES, coverage_fraction, score_regions

# Heuristic reference: a face with this fraction of wrinkle-flagged pixels maps
# to overall_score 1.0. Uncalibrated — tune against labelled data before relying
# on the absolute value (relative ordering is meaningful regardless).
FULL_COVERAGE_REF = 0.08


def normalize_unet_input(masked_rgb: np.ndarray, texture: np.ndarray) -> np.ndarray:
    """Assemble the (1, 4, H, W) float32 [-1, 1] tensor the U-Net expects."""
    img = masked_rgb.astype(np.float32) / 255.0 * 2.0 - 1.0     # (H, W, 3)
    tex = texture.astype(np.float32) / 255.0 * 2.0 - 1.0        # (H, W)
    img_chw = np.transpose(img, (2, 0, 1))                      # (3, H, W)
    tex_chw = tex[None, ...]                                    # (1, H, W)
    combined = np.concatenate([img_chw, tex_chw], axis=0)       # (4, H, W)
    return combined[None, ...].astype(np.float32)               # (1, 4, H, W)


def logits_to_mask(logits: np.ndarray) -> np.ndarray:
    """(1, 2, H, W) class logits -> uint8 (H, W) binary mask (0/255)."""
    pred = np.asarray(logits)
    if pred.ndim == 4:
        pred = pred[0]
    cls = pred.argmax(axis=0)
    return (cls > 0).astype(np.uint8) * 255


@dataclass
class WrinkleResult:
    overall_score: float
    coverage_fraction: float
    regions: dict[str, float]
    mask_available: bool
    detected: bool
    mask: np.ndarray | None = field(default=None, repr=False)
    # Debug intermediates (populated when keep_mask=True) — for overlay rendering.
    crop_rgb: np.ndarray | None = field(default=None, repr=False)
    masked_rgb: np.ndarray | None = field(default=None, repr=False)
    texture: np.ndarray | None = field(default=None, repr=False)
    face_mask: np.ndarray | None = field(default=None, repr=False)
    landmarks_crop: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": round(float(self.overall_score), 4),
            "coverage_fraction": round(float(self.coverage_fraction), 5),
            "regions": {k: round(float(v), 4) for k, v in self.regions.items()},
            "mask_available": bool(self.mask_available),
        }


class WrinkleAnalyzer:
    """
    Loads the wrinkle U-Net ONNX once; call `.analyze()` per selfie.

    Args:
        onnx_path: path to wrinkle_unet.onnx.
        providers: onnxruntime providers (default CPU).
        session:   inject a preconstructed session / stub (used by tests).
    """

    def __init__(self, onnx_path: str | None = None, providers=None, session: Any = None):
        if session is not None:
            self.session = session
        else:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                str(onnx_path), providers=providers or ["CPUExecutionProvider"]
            )
        self._input_name = self.session.get_inputs()[0].name

    def run_mask(self, masked_rgb: np.ndarray, texture: np.ndarray) -> np.ndarray:
        x = normalize_unet_input(masked_rgb, texture)
        logits = self.session.run(None, {self._input_name: x})[0]
        return logits_to_mask(logits)

    def analyze(
        self,
        rgb: np.ndarray,
        landmarks_px: np.ndarray | None = None,
        keep_mask: bool = False,
    ) -> WrinkleResult:
        prep = prepare_wrinkle_input(rgb, landmarks_px=landmarks_px, output_size=OUTPUT_SIZE)
        mask = self.run_mask(prep.masked_rgb, prep.texture)

        lm_crop = None
        if landmarks_px is not None:
            lm_crop = prep.transform.apply(np.asarray(landmarks_px, np.float32))

        cov = coverage_fraction(mask, prep.face_mask)
        regions = score_regions(mask, lm_crop, prep.face_mask)
        overall = float(np.clip(cov / FULL_COVERAGE_REF, 0.0, 1.0))

        return WrinkleResult(
            overall_score=overall,
            coverage_fraction=cov,
            regions=regions,
            mask_available=True,
            detected=prep.detected,
            mask=mask if keep_mask else None,
            crop_rgb=prep.crop_rgb if keep_mask else None,
            masked_rgb=prep.masked_rgb if keep_mask else None,
            texture=prep.texture if keep_mask else None,
            face_mask=prep.face_mask if keep_mask else None,
            landmarks_crop=lm_crop if keep_mask else None,
        )


def empty_wrinkle_result() -> dict[str, Any]:
    """Wrinkle block used when the branch cannot run (no model / no face)."""
    return {
        "overall_score": 0.0,
        "coverage_fraction": 0.0,
        "regions": {name: 0.0 for name in REGION_NAMES},
        "mask_available": False,
    }
