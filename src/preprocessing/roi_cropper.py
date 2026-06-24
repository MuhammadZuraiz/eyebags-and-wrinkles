#!/usr/bin/env python3
"""
Under-eye ROI (Region of Interest) cropper for DermaLens.

What this module does:
  Takes a full face image and the landmarks from FaceLandmarker, then extracts
  two standardised 256×160 crops — one for the left eye area, one for the right.
  These crops are what the model sees during training and inference.

Why 256×160?
  Eye-bag puffiness is wider than it is tall. The 1.6:1 aspect ratio matches
  the natural anatomy while giving the encoder enough horizontal resolution.

Orientation convention:
  Both crops are stored with the OUTER (temporal) eye corner on the LEFT side.
  This means the RIGHT eye crop is flipped horizontally after extraction.
  Consistent orientation lets the model's encoder learn a single representation
  instead of two mirror-image ones.

Usage:
    import cv2
    from src.preprocessing.face_landmarks import FaceLandmarker
    from src.preprocessing.roi_cropper import RoiCropper

    landmarker = FaceLandmarker()
    cropper    = RoiCropper()
    image      = cv2.imread("selfie.jpg")

    lm   = landmarker.detect(image)
    rois = cropper.crop(image, lm)

    if rois.both_valid:
        cv2.imwrite("left_eye_roi.jpg",  rois.left.crop)
        cv2.imwrite("right_eye_roi.jpg", rois.right.crop)
    else:
        print("Left:", rois.left.error_msg)
        print("Right:", rois.right.error_msg)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.preprocessing.face_landmarks import FaceLandmarks

logger = logging.getLogger(__name__)

# ── Output dimensions (never change these without retraining everything) ────
ROI_W = 256
ROI_H = 160

# ── ROI expansion factors (relative to the measured eye width) ──────────────
# MARGIN_HORIZ:  how much to extend BEYOND each eye corner horizontally.
#                0.20 = 20% of eye_width on each side.
MARGIN_HORIZ  = 0.20

# MARGIN_TOP:    how far ABOVE the lower eyelid the crop starts.
#                We include a sliver of the eyelid itself as an anatomical reference.
MARGIN_TOP    = 0.20

# MARGIN_BOTTOM: how far BELOW the lower eyelid the crop extends.
#                1.10 = 110% of eye_width below the lid — enough to capture grade-4 bags.
MARGIN_BOTTOM = 1.10

# ── Minimum eye width in pixels to accept a crop ───────────────────────────
MIN_EYE_WIDTH_PX = 20


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CropResult:
    """
    Under-eye crop for one eye.

    Attributes:
        success:   True if a valid crop was produced.
        crop:      uint8 BGR array, shape (ROI_H, ROI_W, 3) = (160, 256, 3).
                   None if success=False.
        bbox_xyxy: Bounding box in the ORIGINAL image, (x1, y1, x2, y2).
        flipped:   True if the crop was horizontally flipped (right eye).
        error_msg: Human-readable failure reason.
    """
    success:   bool
    crop:      Optional[np.ndarray]                  = None
    bbox_xyxy: Optional[Tuple[int, int, int, int]]   = None
    flipped:   bool                                  = False
    error_msg: str                                   = ""


@dataclass
class RoiPair:
    """
    Both under-eye crops for one face image.

    Attributes:
        left:       CropResult for the left-eye area (image-perspective left).
        right:      CropResult for the right-eye area (image-perspective right).
        both_valid: Convenience property — True only if BOTH crops succeeded.
    """
    left:  CropResult
    right: CropResult

    @property
    def both_valid(self) -> bool:
        return self.left.success and self.right.success


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class RoiCropper:
    """
    Crops under-eye ROIs from a full-face image given MediaPipe landmarks.

    Stateless — no model weights, no internal state. Safe to share between threads.
    """

    def crop(self, image_bgr: np.ndarray, landmarks: FaceLandmarks) -> RoiPair:
        """
        Extract left and right under-eye crops from a full-face image.

        Args:
            image_bgr:  Full face image, shape (H, W, 3), uint8, BGR.
            landmarks:  FaceLandmarks from FaceLandmarker.detect() (normalised coords).

        Returns:
            RoiPair with .left and .right CropResult objects.
            Always returns a RoiPair — check .success on each side.
        """
        if not landmarks.success:
            fail = CropResult(success=False, error_msg="Landmark detection failed upstream")
            return RoiPair(left=fail, right=fail)

        if image_bgr is None or image_bgr.ndim != 3:
            fail = CropResult(success=False, error_msg="Invalid image array passed to RoiCropper")
            return RoiPair(left=fail, right=fail)

        # Convert landmark coordinates from normalised [0,1] to actual pixels
        px = landmarks.to_pixel()
        img_h, img_w = image_bgr.shape[:2]

        left  = self._crop_eye(image_bgr, px, side="left",  img_h=img_h, img_w=img_w)
        right = self._crop_eye(image_bgr, px, side="right", img_h=img_h, img_w=img_w)

        return RoiPair(left=left, right=right)

    # ─────────────────────────────────────────────────────────────────────────

    def _crop_eye(
        self,
        image_bgr: np.ndarray,
        px: FaceLandmarks,   # already converted to pixel coords
        side: str,           # "left" or "right"
        img_h: int,
        img_w: int,
    ) -> CropResult:
        """
        Extract the under-eye crop for one side.

        Steps:
        1. Select the correct landmark arrays for this side.
        2. Measure the eye width from the corner landmarks.
        3. Build a bounding box: eye-width-proportional margins on all sides.
        4. Clamp the box to the image boundary.
        5. Crop and resize to (ROI_W, ROI_H) = (256, 160).
        6. Flip horizontally if it's the right eye (for orientation consistency).
        """

        # ── Step 1: pick the correct side's landmarks ──────────────────────
        if side == "left":
            outer_pt   = px.left_outer
            inner_pt   = px.left_inner
            lower_pts  = px.left_lower_pts
            cheek_pts  = px.left_cheek
            do_flip    = False   # Left eye: outer corner already on image-left → correct
        else:
            outer_pt   = px.right_outer
            inner_pt   = px.right_inner
            lower_pts  = px.right_lower_pts
            cheek_pts  = px.right_cheek
            do_flip    = True    # Right eye: outer corner on image-right → flip after crop

        # ── Step 2: validate required landmarks ───────────────────────────
        missing = [
            name for name, arr in [
                ("outer_corner", outer_pt),
                ("inner_corner", inner_pt),
                ("lower_lid",    lower_pts),
            ]
            if arr is None
        ]
        if missing:
            return CropResult(
                success=False,
                error_msg=f"{side} eye: missing landmarks {missing}"
            )

        # ── Step 3: measure eye width ──────────────────────────────────────
        # eye_width is the horizontal distance between the two eye corners.
        # All margins are expressed as multiples of this width.
        eye_width = abs(float(inner_pt[0]) - float(outer_pt[0]))

        if eye_width < MIN_EYE_WIDTH_PX:
            return CropResult(
                success=False,
                error_msg=(
                    f"{side} eye width = {eye_width:.0f}px < {MIN_EYE_WIDTH_PX}px minimum. "
                    f"Face is likely too far from the camera."
                )
            )

        # ── Step 4: define the bounding box ───────────────────────────────
        # TOP boundary: lower_pts[:,1].min() is the y-coordinate of the highest
        # point of the lower eyelid. We go a bit above it to include the lid.
        lower_lid_y = float(lower_pts[:, 1].min())

        # BOTTOM boundary: use cheek landmarks if available, else estimate.
        if cheek_pts is not None and len(cheek_pts) > 0:
            cheek_y = float(cheek_pts[:, 1].max())
            # Sanity check: cheek should be below the lower lid
            if cheek_y <= lower_lid_y:
                cheek_y = lower_lid_y + eye_width * MARGIN_BOTTOM
        else:
            cheek_y = lower_lid_y + eye_width * MARGIN_BOTTOM

        # Collect all x-coordinates to determine horizontal span
        all_x = np.concatenate([
            lower_pts[:, 0],
            [float(outer_pt[0]), float(inner_pt[0])],
        ])

        x1_f = all_x.min() - eye_width * MARGIN_HORIZ
        x2_f = all_x.max() + eye_width * MARGIN_HORIZ
        y1_f = lower_lid_y - eye_width * MARGIN_TOP
        y2_f = cheek_y     + eye_width * 0.05          # small extra buffer at bottom

        # ── Step 5: clamp to image bounds ─────────────────────────────────
        x1 = max(0, int(round(x1_f)))
        y1 = max(0, int(round(y1_f)))
        x2 = min(img_w - 1, int(round(x2_f)))
        y2 = min(img_h - 1, int(round(y2_f)))

        # Check the resulting crop is large enough to be meaningful
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w < 10 or crop_h < 5:
            return CropResult(
                success=False,
                error_msg=(
                    f"{side} eye ROI collapsed to {crop_w}×{crop_h}px after clamping. "
                    f"Face may be near the image border."
                )
            )

        # ── Step 6: crop and resize ───────────────────────────────────────
        crop = image_bgr[y1:y2, x1:x2]

        # INTER_LINEAR is fast and appropriate for downscaling to 256×160
        crop_resized = cv2.resize(crop, (ROI_W, ROI_H), interpolation=cv2.INTER_LINEAR)

        # ── Step 7: normalise orientation ─────────────────────────────────
        # Convention: in every crop, the OUTER (temporal) corner is on the LEFT.
        # Left eye: outer corner is image-left already → no flip needed.
        # Right eye: outer corner is image-right → flip horizontally.
        if do_flip:
            crop_resized = cv2.flip(crop_resized, flipCode=1)  # 1 = horizontal flip

        return CropResult(
            success=True,
            crop=crop_resized,
            bbox_xyxy=(x1, y1, x2, y2),
            flipped=do_flip,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Quick visual check helper  (used by batch_crop.py for spot-checking)
# ──────────────────────────────────────────────────────────────────────────────

def draw_roi_overlay(
    image_bgr: np.ndarray,
    rois: RoiPair,
    color_left:  Tuple[int, int, int] = (0, 200, 0),
    color_right: Tuple[int, int, int] = (0, 100, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw the ROI bounding boxes onto a copy of the original image.

    Useful for the Day 2 spot-check:
        overlay = draw_roi_overlay(image, rois)
        cv2.imwrite("check_rois.jpg", overlay)
    """
    out = image_bgr.copy()

    for crop_result, color in [(rois.left, color_left), (rois.right, color_right)]:
        if crop_result.success and crop_result.bbox_xyxy is not None:
            x1, y1, x2, y2 = crop_result.bbox_xyxy
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

    return out
