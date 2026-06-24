#!/usr/bin/env python3
"""
Overlay rendering for the verification frontend.

Turns the debug intermediates from FaceSkinAnalyzer.analyze_debug() into a set
of human-inspectable overlay images (returned as PNG data-URIs):

  - eye_bag_overlay : original face with the two under-eye ROI boxes the model
                      actually saw, labelled with grade / probability.
  - eye_crops       : the two 256x160 crops fed to the eye-bag model.
  - wrinkle_overlay : the U-Net's wrinkle segmentation mask painted over the face
                      (exactly which pixels it flagged).
  - wrinkle_regions : the anatomical region polygons used for per-region scores.
  - texture         : the high-pass texture map (the U-Net's 4th input channel).

All work in RGB uint8; OpenCV is used purely as a drawing/encoding tool.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, Optional

import cv2
import numpy as np

from src.wrinkles.regions import region_polygons

# Severity grade -> RGB (green = none ... red = very pronounced).
GRADE_COLORS = {
    0: (60, 200, 90), 1: (170, 210, 60), 2: (240, 200, 50),
    3: (240, 140, 40), 4: (230, 60, 50),
}
WRINKLE_COLOR = (240, 60, 60)      # red mask paint
REGION_COLOR = (80, 180, 255)      # cyan-ish region outline
MAX_W = 768                        # cap overlay width for the web payload


def encode_png_b64(rgb: np.ndarray) -> str:
    """RGB uint8 -> 'data:image/png;base64,...'."""
    rgb = _fit_width(np.ascontiguousarray(rgb))
    ok, buf = cv2.imencode(".png", rgb[:, :, ::-1])   # RGB -> BGR for cv2
    if not ok:
        raise RuntimeError("PNG encode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def render_overlays(debug: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Build every available overlay from a debug bundle. Missing pieces are skipped."""
    out: Dict[str, str] = {}
    if not debug:
        return out

    image_bgr = debug.get("image_bgr")
    image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1]) if image_bgr is not None else None

    eb = debug.get("eye_bags")
    if image_rgb is not None and eb:
        out["eye_bag_overlay"] = encode_png_b64(draw_eye_bag_overlay(image_rgb, eb))
        crops = eye_crops_panel(eb)
        if crops is not None:
            out["eye_crops"] = encode_png_b64(crops)

    wr = debug.get("wrinkles")
    if wr:
        mo = wrinkle_mask_overlay(wr)
        if mo is not None:
            out["wrinkle_overlay"] = encode_png_b64(mo)
        ro = wrinkle_regions_overlay(wr)
        if ro is not None:
            out["wrinkle_regions"] = encode_png_b64(ro)
        if wr.get("texture") is not None:
            out["texture"] = encode_png_b64(_gray_to_rgb(wr["texture"]))
    return out


# ── eye bags ────────────────────────────────────────────────────────────────

def draw_eye_bag_overlay(image_rgb: np.ndarray, eb: Dict[str, Any]) -> np.ndarray:
    out = image_rgb.copy()
    for side in ("left", "right"):
        bbox = eb.get(f"{side}_bbox")
        res = eb.get(side)
        if bbox is None or res is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = GRADE_COLORS.get(res["severity_grade"], (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, max(2, out.shape[1] // 400))
        label = f"{side} g{res['severity_grade']} p{res['present_probability']:.2f}"
        _label(out, label, (x1, y1 - 6), color)
    return out


def eye_crops_panel(eb: Dict[str, Any]) -> Optional[np.ndarray]:
    tiles = []
    for side in ("left", "right"):
        crop = eb.get(f"{side}_crop")
        res = eb.get(side)
        if crop is None:
            continue
        tile = np.ascontiguousarray(crop[:, :, ::-1])   # BGR crop -> RGB
        tile = cv2.resize(tile, (384, 240), interpolation=cv2.INTER_NEAREST)
        if res is not None:
            color = GRADE_COLORS.get(res["severity_grade"], (255, 255, 255))
            cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), color, 4)
            _label(tile, f"{side}: {res['severity_label']} ({res['confidence']:.2f})",
                   (6, 22), color)
        tiles.append(tile)
    if not tiles:
        return None
    gap = np.full((tiles[0].shape[0], 12, 3), 20, np.uint8)
    row = tiles[0] if len(tiles) == 1 else np.hstack([tiles[0], gap, tiles[1]])
    return row


# ── wrinkles ────────────────────────────────────────────────────────────────

def wrinkle_mask_overlay(wr: Dict[str, Any], alpha: float = 0.55) -> Optional[np.ndarray]:
    base = wr.get("crop_rgb")
    mask = wr.get("mask")
    if base is None or mask is None:
        return None
    out = base.copy()
    sel = mask > 0
    paint = np.zeros_like(out)
    paint[:] = WRINKLE_COLOR
    out[sel] = (out[sel] * (1 - alpha) + paint[sel] * alpha).astype(np.uint8)
    return out


def wrinkle_regions_overlay(wr: Dict[str, Any]) -> Optional[np.ndarray]:
    base = wr.get("crop_rgb")
    lm = wr.get("landmarks_crop")
    regions = wr.get("regions", {})
    if base is None or lm is None:
        return None
    out = base.copy()
    polys = region_polygons(lm)
    overlay = out.copy()
    for name, poly in polys.items():
        cv2.fillConvexPoly(overlay, poly, REGION_COLOR)
        cv2.polylines(out, [poly], True, REGION_COLOR, 2)
        cx, cy = poly.mean(axis=0).astype(int)
        _label(out, f"{name} {regions.get(name, 0.0):.3f}", (int(cx) - 40, int(cy)),
               REGION_COLOR, scale=0.5)
    return cv2.addWeighted(overlay, 0.20, out, 0.80, 0)


# ── helpers ───────────────────────────────────────────────────────────────────

def _label(img, text, org, color, scale: float = 0.6):
    x, y = int(org[0]), int(org[1])
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + bl), (15, 15, 20), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    g = np.ascontiguousarray(gray)
    if g.ndim == 3:
        return g
    return np.stack([g, g, g], axis=-1)


def _fit_width(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    if w <= MAX_W:
        return rgb
    s = MAX_W / w
    return cv2.resize(rgb, (MAX_W, int(round(h * s))), interpolation=cv2.INTER_AREA)
