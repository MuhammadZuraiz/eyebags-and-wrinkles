#!/usr/bin/env python3
"""
MediaPipe face landmark extraction for DermaLens under-eye detection.

What this module does:
  Wraps Google's MediaPipe FaceMesh to detect 468 facial landmarks in an image
  and extract the specific anatomical points that define the under-eye (infraorbital)
  region. Those points are then used by roi_cropper.py to slice the exact area
  where eye bags appear.

Why we use MediaPipe instead of a custom detector:
  MediaPipe Face Mesh runs on-device, is fast enough for mobile, and gives us
  precise sub-pixel landmark coordinates. We get 468 points per face, and we
  only need about 30 of them — the ones around and below the lower eyelid.

Usage:
    import cv2
    from src.preprocessing.face_landmarks import FaceLandmarker

    landmarker = FaceLandmarker()
    image = cv2.imread("selfie.jpg")               # BGR uint8
    result = landmarker.detect(image)

    if result.success:
        pixel = result.to_pixel()
        print("Left outer corner:", pixel.left_outer)   # [x, y] in pixels
    else:
        print("Detection failed:", result.error_msg)
"""

import copy
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Landmark index groups
# These refer to the 468-point MediaPipe FaceMesh topology.
# Visualisation: https://github.com/google/mediapipe/blob/master/mediapipe/modules/
#                face_geometry/data/canonical_face_model_uv_visualization.png
# NOTE: "left" and "right" are from the IMAGE perspective, not the subject's.
# ──────────────────────────────────────────────────────────────────────────────

# Left eye from the camera's point of view (= person's right eye)
LEFT_EYE_OUTER    = 33     # Outer (temporal) corner
LEFT_EYE_INNER    = 133    # Inner (nasal) corner
LEFT_LOWER_LID    = [145, 153, 154, 155]        # Lower eyelid arc
LEFT_INFRAORBITAL = [226, 31, 228, 229, 230,    # Infraorbital rim — where bags bulge
                     231, 232, 233, 244]
LEFT_CHEEK_REF    = [207, 205, 206, 36, 142]    # Upper cheek — lower ROI boundary

# Right eye from the camera's point of view (= person's left eye)
RIGHT_EYE_OUTER    = 263
RIGHT_EYE_INNER    = 362
RIGHT_LOWER_LID    = [374, 380, 381, 382]
RIGHT_INFRAORBITAL = [446, 261, 448, 449, 450,
                      451, 452, 453, 464]
RIGHT_CHEEK_REF    = [427, 425, 426, 266, 371]


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FaceLandmarks:
    """
    Structured output from face landmark detection.

    All coordinates are NORMALISED to the range [0.0, 1.0] relative to
    image width (x) and height (y). Call `.to_pixel()` to get pixel coords.
    """
    success: bool

    # Image shape (needed for coordinate conversion)
    height: int = 0
    width:  int = 0

    # Eye corner points — shape (2,) each, [x, y] normalised
    left_outer:  Optional[np.ndarray] = None   # temporal corner
    left_inner:  Optional[np.ndarray] = None   # nasal corner
    right_outer: Optional[np.ndarray] = None
    right_inner: Optional[np.ndarray] = None

    # Under-eye boundary points — shape (N, 2), [x, y] normalised
    # These define the upper and lower edges of the ROI crop box.
    left_lower_pts:  Optional[np.ndarray] = None   # lower eyelid + infraorbital
    right_lower_pts: Optional[np.ndarray] = None

    # Cheek reference — defines how far down we extend the ROI
    left_cheek:  Optional[np.ndarray] = None   # shape (M, 2)
    right_cheek: Optional[np.ndarray] = None

    # Quality / pose signals — used by the quality gate
    face_confidence: float = 0.0   # 0–1, higher is better
    pose_yaw_deg:    float = 0.0   # horizontal head rotation (turned left/right)
    pose_pitch_deg:  float = 0.0   # vertical head tilt (up/down)
    pose_roll_deg:   float = 0.0   # in-plane head tilt (ear toward shoulder)

    error_msg: str = ""

    def to_pixel(self) -> "FaceLandmarks":
        """
        Return a copy of this object with all coordinates scaled to pixel space.

        Multiply normalised x by image width, normalised y by image height.
        """
        if not self.success:
            return self

        scale = np.array([self.width, self.height], dtype=np.float32)

        def _scale(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
            return (arr * scale).astype(np.float32) if arr is not None else None

        result = copy.copy(self)
        result.left_lower_pts  = _scale(self.left_lower_pts)
        result.right_lower_pts = _scale(self.right_lower_pts)
        result.left_outer      = _scale(self.left_outer)
        result.left_inner      = _scale(self.left_inner)
        result.right_outer     = _scale(self.right_outer)
        result.right_inner     = _scale(self.right_inner)
        result.left_cheek      = _scale(self.left_cheek)
        result.right_cheek     = _scale(self.right_cheek)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Landmarker
# ──────────────────────────────────────────────────────────────────────────────

class FaceLandmarker:
    """
    Detects face landmarks in a BGR image using MediaPipe FaceMesh.

    Create one instance per process (MediaPipe models are loaded at init time).
    The instance is NOT thread-safe — create one per thread if parallelising.

    Args:
        min_detection_confidence:
            How confident MediaPipe must be before declaring a face found.
            Lower values find more faces but risk false detections.
            0.5 is a good default for selfie-quality images.
        refine_landmarks:
            Whether to use the more precise iris-refine model.
            Adds ~5 ms per image on CPU. Recommended: True.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        refine_landmarks: bool = True,
    ):
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError(
                "mediapipe is not installed.\n"
                "Run: pip install mediapipe>=0.10.0"
            )

        self._mp = mp
        # static_image_mode=True: treat every call as a fresh image (no tracking between frames)
        # max_num_faces=1: we only need to analyse the user's face
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def detect(self, image_bgr: np.ndarray) -> FaceLandmarks:
        """
        Run face landmark detection on a BGR image.

        Args:
            image_bgr:
                numpy array, shape (H, W, 3), dtype uint8.
                Standard OpenCV format (BGR channel order).

        Returns:
            FaceLandmarks with normalised (0–1) coordinates.
            Check .success before using any landmark arrays.
            If .success is False, .error_msg explains why.
        """
        # ── Input validation ──────────────────────────────────────────────
        if image_bgr is None:
            return FaceLandmarks(success=False, error_msg="image_bgr is None")
        if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3:
            return FaceLandmarks(success=False, error_msg="Expected (H,W,3) uint8 array")

        h, w = image_bgr.shape[:2]

        # ── Run MediaPipe (it requires RGB, not BGR) ───────────────────────
        image_rgb = image_bgr[:, :, ::-1]   # Reverse channel order; fastest way in numpy

        try:
            mp_result = self._mesh.process(image_rgb)
        except Exception as exc:
            return FaceLandmarks(
                success=False, height=h, width=w,
                error_msg=f"MediaPipe exception: {exc}"
            )

        if not mp_result.multi_face_landmarks:
            return FaceLandmarks(
                success=False, height=h, width=w,
                error_msg="No face detected"
            )

        # ── Extract all 468+ landmark coordinates ─────────────────────────
        face  = mp_result.multi_face_landmarks[0]
        n_lm  = len(face.landmark)
        # Columns: [x, y] — normalised to image dimensions
        all_pts = np.array(
            [[lm.x, lm.y] for lm in face.landmark],
            dtype=np.float32
        )

        def _get(indices: List[int]) -> Optional[np.ndarray]:
            """Get a subset of landmarks as (N,2) array, ignoring out-of-range indices."""
            valid = [i for i in indices if 0 <= i < n_lm]
            return all_pts[valid] if valid else None

        def _get1(idx: int) -> Optional[np.ndarray]:
            """Get a single landmark as (2,) array."""
            return all_pts[idx].copy() if 0 <= idx < n_lm else None

        # ── Estimate pose roll from eye geometry ──────────────────────────
        # The angle of the line joining the two eye centres measures IN-PLANE
        # rotation (head tilted ear-to-shoulder), i.e. roll — not yaw.
        left_ctr  = all_pts[[LEFT_EYE_OUTER,  LEFT_EYE_INNER]].mean(axis=0)
        right_ctr = all_pts[[RIGHT_EYE_OUTER, RIGHT_EYE_INNER]].mean(axis=0)
        dx = right_ctr[0] - left_ctr[0]
        dy = right_ctr[1] - left_ctr[1]
        roll_deg = float(np.degrees(np.arctan2(dy, dx)))

        nose_tip = all_pts[1] if n_lm > 1 else np.array([0.5, 0.5])
        eye_mid  = (left_ctr + right_ctr) * 0.5

        # ── Estimate pose yaw from nose-to-eye-midpoint offset ────────────
        # When the head turns left/right, the nose tip shifts horizontally
        # relative to the midpoint between the eyes; on a frontal face it sits
        # almost exactly under it. Normalising by interocular distance makes
        # this scale-invariant. Rough calibration: a full interocular width of
        # offset ≈ 90° (in practice rejects kick in far below that).
        interocular = float(np.hypot(dx, dy))
        if interocular > 1e-6:
            yaw_deg = float(np.clip(
                (nose_tip[0] - eye_mid[0]) / interocular * 90.0,
                -60.0, 60.0
            ))
        else:
            yaw_deg = 0.0

        # ── Estimate pose pitch from eye-to-nose relationship ─────────────
        # The vertical nose-to-eye-midpoint distance, normalised by interocular
        # width, is roughly constant (~0.55) on a frontal face regardless of
        # how much of the frame the face fills. Tilting the head up or down
        # shortens/lengthens the projected distance; deviation from the
        # baseline is the pitch proxy. Baseline 0.65 and 0.25-per-45° slope
        # calibrated empirically on the London Set studio frontals (their
        # nose ratios cluster at 0.61-0.71 with heads level).
        FRONTAL_NOSE_RATIO = 0.65
        if interocular > 1e-6:
            nose_ratio = (nose_tip[1] - eye_mid[1]) / interocular
            pitch_deg = float(np.clip(
                (FRONTAL_NOSE_RATIO - nose_ratio) / 0.25 * 45.0,
                -60.0, 60.0
            ))
        else:
            pitch_deg = 0.0

        # ── Rough face confidence from z-coordinate spread ────────────────
        # MediaPipe doesn't expose detection confidence in FaceMesh mode.
        # We use the z-value standard deviation as a proxy: well-detected frontal
        # faces have a predictable z-spread; distorted/side-on faces are noisier.
        z_vals     = np.array([lm.z for lm in face.landmark])
        confidence = float(np.clip(1.0 - np.std(z_vals) * 8.0, 0.1, 1.0))

        # ── Combine lower lid + infraorbital indices for richer ROI boundary ─
        left_lower  = _get(LEFT_LOWER_LID  + LEFT_INFRAORBITAL)
        right_lower = _get(RIGHT_LOWER_LID + RIGHT_INFRAORBITAL)

        return FaceLandmarks(
            success         = True,
            height          = h,
            width           = w,
            left_outer      = _get1(LEFT_EYE_OUTER),
            left_inner      = _get1(LEFT_EYE_INNER),
            right_outer     = _get1(RIGHT_EYE_OUTER),
            right_inner     = _get1(RIGHT_EYE_INNER),
            left_lower_pts  = left_lower,
            right_lower_pts = right_lower,
            left_cheek      = _get(LEFT_CHEEK_REF),
            right_cheek     = _get(RIGHT_CHEEK_REF),
            face_confidence = confidence,
            pose_yaw_deg    = yaw_deg,
            pose_pitch_deg  = pitch_deg,
            pose_roll_deg   = roll_deg,
        )

    def __del__(self):
        """Release the MediaPipe model when this object is garbage-collected."""
        try:
            if hasattr(self, "_mesh") and self._mesh is not None:
                self._mesh.close()
        except Exception:
            pass
