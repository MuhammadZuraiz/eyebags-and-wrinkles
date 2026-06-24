#!/usr/bin/env python3
"""
Image quality gate for DermaLens.

What this module does:
  Runs fast rule-based checks on a face image BEFORE sending it to MediaPipe
  or the ML model. Rejects images that are too blurry, too dark, too small,
  or at the wrong angle. This prevents the model from seeing inputs it was
  never trained on.

Why rule-based first?
  A learned quality model adds complexity and needs its own training data.
  Deterministic checks (Laplacian variance, pixel brightness, MediaPipe pose)
  catch the most common failure modes without any training.

Checks performed (in order):
  1. Resolution: is the image at least 200×200 pixels?
  2. Blur: Laplacian variance above threshold?
  3. Brightness + glare: face region not too dark, not overexposed, no heavy glare?
  4. Face detected: did MediaPipe find a face?
  5. Pose yaw: is the head facing mostly forward (|yaw| < 20°)?
  6. Pose pitch: is the head tilt within range (|pitch| < 25°)?
  7. Pose roll: is the head level, not tilted ear-to-shoulder (|roll| < 20°)?
  8. Face size: does the eye region occupy enough of the image?
  9. Occlusion proxy: are both eye areas visible (not behind sunglasses / cropped)?

Usage:
    import cv2
    from src.preprocessing.quality_gate import QualityGate

    gate = QualityGate()
    image = cv2.imread("selfie.jpg")
    result = gate.check(image)

    if result.accepted:
        # proceed to landmark extraction and cropping
        pass
    else:
        print("Quality issues:", result.reasons)
        # result.retake_message is the user-facing string to display in the app
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Thresholds (all tunable)
# ──────────────────────────────────────────────────────────────────────────────

class QualityThresholds:
    """All thresholds in one place so they're easy to adjust."""

    # Minimum image dimension (narrower side)
    MIN_IMAGE_DIM_PX:      int   = 200

    # Laplacian variance for blur detection, measured on the CENTRAL 60% of
    # the frame (flat backgrounds dilute the whole-image variance — studio
    # portraits scored "blurry" before this). Typical central values:
    # 10 = very blurry, 40 = borderline, 100+ = sharp.
    MIN_BLUR_VARIANCE:     float = 40.0

    # Brightness + glare, measured as Rec.709 luma on the CENTRAL 60% (the face
    # region) — not the whole frame, so dark/bright backgrounds don't fool it.
    # Ported from the selfie-preprocess pipeline (median luma + glare fraction),
    # which caught dark/overexposed selfies the old whole-image mean waved
    # through. Uses the median (robust to a few bright/dark pixels).
    MIN_FACE_LUMINANCE:    float = 45.0     # Too dark   (median luma)
    MAX_FACE_LUMINANCE:    float = 218.0    # Overexposed (median luma)
    MAX_GLARE_FRACTION:    float = 0.03     # Fraction of near-white (>=245) pixels

    # Head pose (degrees)
    MAX_YAW_DEG:           float = 20.0    # Left-right rotation
    MAX_PITCH_DEG:         float = 30.0    # Up-down tilt (proxy has ~±10°
                                           # anatomical noise; verified on
                                           # London Set studio frontals)
    MAX_ROLL_DEG:          float = 20.0    # In-plane tilt (ear toward shoulder)

    # Minimum MediaPipe face confidence
    MIN_FACE_CONFIDENCE:   float = 0.30

    # Eye-size floor. The binding constraint is RESOLUTION — the crop must not
    # be upscaled from a tiny region — so the primary check is pixels. The
    # fraction check is only a sanity floor for faces lost in a huge frame
    # (0.15 was selfie-calibrated and rejected every full-head portrait).
    MIN_EYE_WIDTH_PX:       float = 60.0   # Eye width in pixels
    MIN_EYE_WIDTH_FRACTION: float = 0.04   # Eye width / image width


# ──────────────────────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QualityResult:
    """
    Output from QualityGate.check().

    Attributes:
        accepted:       True = proceed to model inference.
                        False = reject this image.
        reasons:        List of machine-readable failure codes. Empty if accepted.
        retake_message: User-facing sentence to show in the app.
        details:        Dict of numeric measurements (blur score, brightness, etc.)
                        Useful for debugging and monitoring.
    """
    accepted:       bool
    reasons:        List[str]        = field(default_factory=list)
    retake_message: str              = ""
    details:        dict             = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Gate
# ──────────────────────────────────────────────────────────────────────────────

_USER_MESSAGES = {
    "too_small":       "Please move closer to the camera.",
    "too_blurry":      "Please hold the camera steady and ensure the image is in focus.",
    "too_dark":        "Please take the photo in brighter, even lighting.",
    "overexposed":     "Please move away from bright backlighting.",
    "heavy_glare":     "There's too much glare or reflection. Please avoid direct light and shiny skin highlights.",
    "no_face":         "We could not detect a face. Please position your face in the centre of the frame.",
    "pose_yaw":        "Please face the camera directly.",
    "pose_pitch":      "Please keep your head level — not tilted up or down.",
    "pose_roll":       "Please keep your head level — not tilted to the side.",
    "face_too_small":  "Please move closer to the camera so your eyes fill more of the frame.",
    "low_confidence":  "We could not read your face clearly. Please retake in even, natural lighting.",
}

_DEFAULT_RETAKE = (
    "We could not analyse the under-eye area reliably. "
    "Please retake the photo in even, natural lighting with your eyes fully open."
)


class QualityGate:
    """
    Fast rule-based quality gate for DermaLens face images.

    Args:
        thresholds: QualityThresholds instance. Pass a customised one to adjust limits.
        landmarker: Optional FaceLandmarker instance. If None, a new one is created
                    on the first call to check(). Providing an existing landmarker
                    avoids loading the MediaPipe model twice.
    """

    def __init__(
        self,
        thresholds: Optional[QualityThresholds] = None,
        landmarker=None,   # FaceLandmarker (avoid circular import with type hint)
    ):
        self.t = thresholds or QualityThresholds()
        self._landmarker = landmarker   # lazy-init if None

    def _ensure_landmarker(self):
        if self._landmarker is None:
            from src.preprocessing.face_landmarks import FaceLandmarker
            self._landmarker = FaceLandmarker()

    def check(self, image_bgr: np.ndarray) -> QualityResult:
        """
        Run all quality checks on a BGR image.

        Checks are ordered from cheapest (array operations) to most expensive
        (MediaPipe inference). We return early at the first critical failure
        to save compute.

        Args:
            image_bgr: numpy array (H, W, 3) uint8 BGR.

        Returns:
            QualityResult — always returned, never raises.
        """
        reasons:  List[str] = []
        details:  dict      = {}

        # ── Check 1: Null / shape check ───────────────────────────────────
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.dtype != np.uint8:
            return QualityResult(
                accepted=False,
                reasons=["invalid_input"],
                retake_message=_DEFAULT_RETAKE,
            )

        h, w = image_bgr.shape[:2]
        details["image_h"] = h
        details["image_w"] = w

        # ── Check 2: Minimum resolution ──────────────────────────────────
        if min(h, w) < self.t.MIN_IMAGE_DIM_PX:
            reasons.append("too_small")
            details["min_dim"] = min(h, w)

        # ── Check 3: Blur (Laplacian variance, central region) ───────────
        # Laplacian highlights edges. Blurry images have weak edges → low
        # variance. Measured on the central 60% so flat backgrounds (studio
        # portraits) don't drag the score down.
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        y0, y1, x0, x1 = int(h * 0.2), int(h * 0.8), int(w * 0.2), int(w * 0.8)
        ctr = gray[y0:y1, x0:x1]
        ctr_bgr = image_bgr[y0:y1, x0:x1]
        if ctr.size == 0:
            ctr, ctr_bgr = gray, image_bgr
        blur_score = float(cv2.Laplacian(ctr, cv2.CV_64F).var())
        details["blur_laplacian"] = round(blur_score, 1)

        if blur_score < self.t.MIN_BLUR_VARIANCE:
            reasons.append("too_blurry")

        # ── Check 4: Brightness + glare (Rec.709 luma on the face region) ──
        b, g, r = ctr_bgr[..., 0], ctr_bgr[..., 1], ctr_bgr[..., 2]
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
        luma_median   = float(np.median(luma))
        glare_fraction = float((luma >= 245).mean())
        details["luma_median"]    = round(luma_median, 1)
        details["glare_fraction"] = round(glare_fraction, 4)

        if luma_median < self.t.MIN_FACE_LUMINANCE:
            reasons.append("too_dark")
        elif luma_median > self.t.MAX_FACE_LUMINANCE:
            reasons.append("overexposed")
        if glare_fraction > self.t.MAX_GLARE_FRACTION:
            reasons.append("heavy_glare")

        # ── Early exit if we already have failures ────────────────────────
        # No point running MediaPipe on an image that's definitely too blurry/small.
        # BUT: we want to check pose even if blur is borderline.
        if "too_small" in reasons:
            return self._build_result(reasons, details)

        # ── Check 5–8: Require MediaPipe face detection ───────────────────
        self._ensure_landmarker()
        lm = self._landmarker.detect(image_bgr)
        details["face_detected"] = lm.success

        if not lm.success:
            reasons.append("no_face")
            return self._build_result(reasons, details)

        details["face_confidence"] = round(lm.face_confidence, 3)
        details["pose_yaw_deg"]    = round(lm.pose_yaw_deg, 1)
        details["pose_pitch_deg"]  = round(lm.pose_pitch_deg, 1)
        details["pose_roll_deg"]   = round(lm.pose_roll_deg, 1)

        # Low face confidence (not a hard reject — combine with other signals)
        if lm.face_confidence < self.t.MIN_FACE_CONFIDENCE:
            reasons.append("low_confidence")

        # Pose yaw (turned left/right)
        if abs(lm.pose_yaw_deg) > self.t.MAX_YAW_DEG:
            reasons.append("pose_yaw")

        # Pose pitch (up-down)
        if abs(lm.pose_pitch_deg) > self.t.MAX_PITCH_DEG:
            reasons.append("pose_pitch")

        # Pose roll (in-plane tilt)
        if abs(lm.pose_roll_deg) > self.t.MAX_ROLL_DEG:
            reasons.append("pose_roll")

        # Eye-size floor: pixel resolution first, frame fraction as sanity check
        if (lm.left_outer is not None and lm.left_inner is not None and
                lm.right_outer is not None and lm.right_inner is not None):
            left_eye_w  = abs(lm.left_inner[0]  - lm.left_outer[0])
            right_eye_w = abs(lm.right_inner[0] - lm.right_outer[0])
            mean_eye_w_fraction = ((left_eye_w + right_eye_w) / 2.0)  # still normalised
            mean_eye_w_px = mean_eye_w_fraction * w
            details["eye_width_fraction"] = round(float(mean_eye_w_fraction), 3)
            details["eye_width_px"]       = round(float(mean_eye_w_px), 1)

            if (mean_eye_w_px < self.t.MIN_EYE_WIDTH_PX
                    or mean_eye_w_fraction < self.t.MIN_EYE_WIDTH_FRACTION):
                reasons.append("face_too_small")

        return self._build_result(reasons, details)

    # ─────────────────────────────────────────────────────────────────────────

    def _build_result(self, reasons: List[str], details: dict) -> QualityResult:
        """Assemble the final QualityResult from a list of failure codes."""
        if not reasons:
            return QualityResult(accepted=True, details=details)

        # Build user-facing message from the most important failure
        priority_order = [
            "too_small", "no_face", "too_blurry", "too_dark",
            "overexposed", "heavy_glare", "pose_yaw", "pose_pitch", "pose_roll",
            "face_too_small", "low_confidence",
        ]
        primary_reason = next(
            (r for r in priority_order if r in reasons),
            reasons[0]
        )
        user_msg = _USER_MESSAGES.get(primary_reason, _DEFAULT_RETAKE)

        return QualityResult(
            accepted=False,
            reasons=reasons,
            retake_message=user_msg,
            details=details,
        )

    def check_batch(self, images: list) -> List[QualityResult]:
        """
        Run quality checks on a list of images.
        Returns one QualityResult per image.
        """
        return [self.check(img) for img in images]
