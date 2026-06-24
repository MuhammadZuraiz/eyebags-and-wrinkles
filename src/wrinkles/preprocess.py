#!/usr/bin/env python3
"""
Face crop + background masking for the wrinkle branch.

Produces the 1024x1024 inputs the vendored U-Net expects:
  - masked RGB face (background zeroed)
  - the high-pass texture map (4th channel)
  - the binary face mask used for masking and for region scoring

Two paths:
  - PREFERRED: landmark-driven. Given the shared MediaPipe face-oval landmarks,
    crop tightly to the face bounding box and build an accurate oval mask. This
    is strictly better than the notebook's fixed ellipse.
  - FALLBACK: no landmarks. Center-square crop + a fixed tight ellipse mask
    (ported from the wrinkle notebook's `make_tight_face_oval_mask`).

Pure numpy + OpenCV (torch-free) so it runs in the on-device path. The returned
``CropTransform`` lets callers map original-image pixel coordinates into the
1024 crop frame, which the region-scoring step needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .texture import generate_texture_map_from_masked_face

# MediaPipe FaceMesh "face oval" landmark indices (468/478 topology) — the same
# set used by selfie_face_preprocess.geometry.FACE_OVAL. Kept local so this
# module has no cross-package import at runtime.
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

OUTPUT_SIZE = 1024


@dataclass
class CropTransform:
    """Maps ORIGINAL-image pixel coords into the square 1024 crop frame.

    crop_xy = (orig_xy - [origin_x, origin_y]) * scale
    """

    origin_x: float
    origin_y: float
    scale: float

    def apply(self, points_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_px, dtype=np.float32)
        return (pts - np.array([self.origin_x, self.origin_y], np.float32)) * self.scale


@dataclass
class WrinkleInput:
    masked_rgb: np.ndarray   # uint8 (1024, 1024, 3), background zeroed
    texture: np.ndarray      # uint8 (1024, 1024)
    face_mask: np.ndarray    # uint8 (1024, 1024), 0/255
    transform: CropTransform
    detected: bool           # True if a landmark-driven crop was used
    crop_rgb: np.ndarray | None = None   # uint8 (1024,1024,3) UNMASKED crop (for overlays)


def prepare_wrinkle_input(
    rgb: np.ndarray,
    landmarks_px: np.ndarray | None = None,
    output_size: int = OUTPUT_SIZE,
    margin: float = 0.12,
    mask_blur: int = 21,
) -> WrinkleInput:
    """
    Build the masked face + texture map for the wrinkle U-Net.

    Args:
        rgb: uint8 (H, W, 3) RGB image (the full selfie).
        landmarks_px: optional (N, 2) array of face landmarks in PIXEL coords of
            ``rgb`` (N >= max(FACE_OVAL)+1). When provided, used for an accurate
            crop + oval mask. When None, falls back to a center crop + ellipse.
        output_size: square output side (default 1024 — do not change without
            re-checking the U-Net's expected input resolution).
        margin: fraction of the face box added on each side before cropping.
        mask_blur: odd Gaussian kernel to feather the mask edge.

    Returns:
        WrinkleInput with masked_rgb, texture, face_mask, transform, detected.
    """
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("prepare_wrinkle_input expects an (H, W, 3) RGB array")

    if landmarks_px is not None and len(landmarks_px) > max(FACE_OVAL):
        return _landmark_crop(rgb, np.asarray(landmarks_px, np.float32),
                              output_size, margin, mask_blur)
    return _fallback_crop(rgb, output_size, mask_blur)


# ── landmark-driven path ────────────────────────────────────────────────────

def _landmark_crop(rgb, landmarks_px, output_size, margin, mask_blur) -> WrinkleInput:
    oval = landmarks_px[FACE_OVAL]                 # (36, 2) pixel coords
    min_xy = oval.min(axis=0)
    max_xy = oval.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    side = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]))
    side *= (1.0 + 2.0 * margin)
    side = max(side, 8.0)

    ox = float(center[0] - side / 2.0)
    oy = float(center[1] - side / 2.0)

    crop = _crop_square_with_pad(rgb, ox, oy, side)
    crop_resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
    scale = output_size / side
    transform = CropTransform(origin_x=ox, origin_y=oy, scale=scale)

    # Oval mask from the oval landmarks mapped into the crop frame.
    oval_crop = transform.apply(oval)
    mask = np.zeros((output_size, output_size), dtype=np.uint8)
    cv2.fillPoly(mask, [oval_crop.astype(np.int32)], 255)
    if mask_blur and mask_blur >= 3:
        k = mask_blur | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        mask = np.where(mask >= 128, 255, 0).astype(np.uint8)

    masked_rgb = _apply_mask(crop_resized, mask)
    texture = generate_texture_map_from_masked_face(masked_rgb, mask)
    return WrinkleInput(masked_rgb, texture, mask, transform, detected=True, crop_rgb=crop_resized)


# ── fallback path (ported from the wrinkle notebook) ────────────────────────

def _fallback_crop(rgb, output_size, mask_blur) -> WrinkleInput:
    h, w = rgb.shape[:2]
    side = min(h, w)
    ox = (w - side) / 2.0
    oy = (h - side) / 2.0
    crop = rgb[int(oy):int(oy) + side, int(ox):int(ox) + side]
    crop_resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
    transform = CropTransform(origin_x=ox, origin_y=oy, scale=output_size / side)

    mask = _make_tight_face_oval_mask(output_size)
    masked_rgb = _apply_mask(crop_resized, mask)
    texture = generate_texture_map_from_masked_face(masked_rgb, mask)
    return WrinkleInput(masked_rgb, texture, mask, transform, detected=False, crop_rgb=crop_resized)


def _make_tight_face_oval_mask(size=1024, rx_ratio=0.30, ry_ratio=0.40, cy_ratio=0.50):
    """Fixed central-face ellipse mask (ported from the wrinkle notebook)."""
    mask = np.zeros((size, size), dtype=np.uint8)
    cx = size // 2
    cy = int(size * cy_ratio)
    rx = int(size * rx_ratio)
    ry = int(size * ry_ratio)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, thickness=-1)
    mask = cv2.GaussianBlur(mask, (41, 41), 0)
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


# ── helpers ─────────────────────────────────────────────────────────────────

def _apply_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = (mask.astype(np.float32) / 255.0)[..., None]
    return np.clip(rgb.astype(np.float32) * m, 0, 255).astype(np.uint8)


def _crop_square_with_pad(rgb: np.ndarray, ox: float, oy: float, side: float) -> np.ndarray:
    """Crop a square starting at (ox, oy) of size `side`, zero-padding out-of-bounds."""
    h, w = rgb.shape[:2]
    side_i = int(round(side))
    ox_i = int(round(ox))
    oy_i = int(round(oy))
    out = np.zeros((side_i, side_i, 3), dtype=rgb.dtype)

    src_x1 = max(0, ox_i)
    src_y1 = max(0, oy_i)
    src_x2 = min(w, ox_i + side_i)
    src_y2 = min(h, oy_i + side_i)
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return out

    dst_x1 = src_x1 - ox_i
    dst_y1 = src_y1 - oy_i
    out[dst_y1:dst_y1 + (src_y2 - src_y1), dst_x1:dst_x1 + (src_x2 - src_x1)] = \
        rgb[src_y1:src_y2, src_x1:src_x2]
    return out
