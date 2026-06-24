#!/usr/bin/env python3
"""
Per-region wrinkle scoring from a segmentation mask + face landmarks.

Given the U-Net's binary wrinkle mask (in the 1024 crop frame) and the face
landmarks mapped into that same frame, we carve the canonical cosmetic wrinkle
zones and report, for each, the fraction of region pixels flagged as wrinkle.

Region polygons are convex hulls of MediaPipe FaceMesh landmark groups (468/478
topology, "left"/"right" = IMAGE perspective). They are deliberately approximate
anatomical zones — good enough for relative, per-region reporting; they are not
clinical boundaries. Pure numpy + OpenCV (torch-free).
"""

from __future__ import annotations

import cv2
import numpy as np

# Landmark index groups whose convex hull defines each region.
REGION_LANDMARKS: dict[str, list[int]] = {
    # Forehead band: oval hairline points (top) down to the eyebrow tops.
    "forehead": [10, 67, 109, 338, 297, 332, 103, 104, 105, 66, 107, 336, 296, 334, 9],
    # Glabella: the "11" frown-line zone between the brows.
    "glabella": [9, 8, 168, 6, 107, 336, 55, 285, 66, 296],
    # Periocular (crow's feet + under-eye) — image-left eye.
    "periocular_left": [33, 133, 7, 163, 144, 145, 153, 154, 155, 246, 161, 160,
                        159, 158, 157, 173, 226, 31, 228, 229, 230, 231, 232, 233, 244],
    # Periocular — image-right eye.
    "periocular_right": [263, 362, 249, 390, 373, 374, 380, 381, 382, 466, 388, 387,
                         386, 385, 384, 398, 446, 261, 448, 449, 450, 451, 452, 453, 464],
    # Nasolabial fold — image-left (nose ala down to mouth corner).
    "nasolabial_left": [129, 98, 205, 50, 187, 147, 123, 116, 61, 91, 212, 57],
    # Nasolabial fold — image-right.
    "nasolabial_right": [358, 327, 425, 280, 411, 376, 352, 345, 291, 321, 432, 287],
}

REGION_NAMES = list(REGION_LANDMARKS.keys())


def _region_polygon(landmarks_crop: np.ndarray, indices: list[int]) -> np.ndarray | None:
    valid = [i for i in indices if 0 <= i < len(landmarks_crop)]
    if len(valid) < 3:
        return None
    pts = landmarks_crop[valid].astype(np.int32)
    hull = cv2.convexHull(pts)
    return hull.reshape(-1, 2)


def region_polygons(landmarks_crop: np.ndarray | None) -> dict[str, np.ndarray]:
    """Convex-hull polygon (Kx2 int) per region, in the crop frame. For overlays."""
    out: dict[str, np.ndarray] = {}
    if landmarks_crop is None:
        return out
    for name, indices in REGION_LANDMARKS.items():
        poly = _region_polygon(landmarks_crop, indices)
        if poly is not None:
            out[name] = poly
    return out


def score_regions(
    wrinkle_mask: np.ndarray,
    landmarks_crop: np.ndarray | None,
    face_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Fraction of each region's pixels flagged as wrinkle.

    Args:
        wrinkle_mask: uint8 (S, S) binary mask (0/255) in the crop frame.
        landmarks_crop: (N, 2) landmarks mapped into the same crop frame, or None.
        face_mask: optional uint8 (S, S) mask to intersect regions with the face.

    Returns:
        {region_name: fraction in [0, 1]} for every region. Regions whose
        landmarks are unavailable report 0.0.
    """
    size = wrinkle_mask.shape[0]
    wr = wrinkle_mask > 0
    out: dict[str, float] = {name: 0.0 for name in REGION_NAMES}
    if landmarks_crop is None:
        return out

    face_bool = (face_mask > 0) if face_mask is not None else None
    for name, indices in REGION_LANDMARKS.items():
        poly = _region_polygon(landmarks_crop, indices)
        if poly is None:
            continue
        region = np.zeros((size, size), dtype=np.uint8)
        cv2.fillConvexPoly(region, poly, 1)
        region_bool = region > 0
        if face_bool is not None:
            region_bool &= face_bool
        denom = int(region_bool.sum())
        if denom == 0:
            continue
        out[name] = float((wr & region_bool).sum()) / float(denom)
    return out


def coverage_fraction(wrinkle_mask: np.ndarray, face_mask: np.ndarray | None) -> float:
    """Fraction of face pixels (or whole frame if no mask) flagged as wrinkle."""
    wr = wrinkle_mask > 0
    if face_mask is not None:
        face = face_mask > 0
        denom = int(face.sum())
        if denom == 0:
            return 0.0
        return float((wr & face).sum()) / float(denom)
    return float(wr.mean())
