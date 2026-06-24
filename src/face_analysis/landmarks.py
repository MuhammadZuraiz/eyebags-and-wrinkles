#!/usr/bin/env python3
"""
Single shared MediaPipe **Tasks** Face Landmarker front-end.

The eye-bags project historically used the legacy `mp.solutions.face_mesh` API
(pinned mediapipe==0.10.14). The selfie-face-preprocess package uses the current
MediaPipe **Tasks** Face Landmarker. To run ONE landmark pass for the whole
unified pipeline — and drop the legacy dependency — this adapter:

  * runs the Tasks landmarker (via selfie_face_preprocess) once per image,
  * exposes a `.detect(image_bgr) -> FaceLandmarks` method that is drop-in
    compatible with the eye-bags `FaceLandmarker`, so `QualityGate` and
    `RoiCropper` are reused UNCHANGED, and
  * caches the last result so QualityGate + RoiCropper + the wrinkle branch all
    share the same single inference (memoised by array identity/shape).

The 478-point Tasks topology shares indices 0..467 with the 468-point FaceMesh
topology the eye-bags index constants were written against, so reusing them is
safe (the extra iris points are appended at the end).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from src.preprocessing.face_landmarks import (
    FaceLandmarks,
    LEFT_CHEEK_REF, LEFT_EYE_INNER, LEFT_EYE_OUTER, LEFT_INFRAORBITAL, LEFT_LOWER_LID,
    RIGHT_CHEEK_REF, RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_INFRAORBITAL, RIGHT_LOWER_LID,
)

logger = logging.getLogger(__name__)


def pose_from_transform_matrix(matrix) -> tuple[float, float, float]:
    """
    Decompose MediaPipe's 4x4 facial transformation matrix into
    (yaw, pitch, roll) degrees.

    This is the RELIABLE pose source for the Tasks landmarker. The eye-bags
    nose-offset / z-std heuristics in :func:`face_landmarks_from_array` were
    calibrated for the legacy FaceMesh `solutions` API and do not transfer to
    the Tasks landmarker (they systematically over-estimate pitch and under-
    estimate confidence). The quality gate only uses ``abs()`` of each angle, so
    the axis sign convention does not matter here.
    """
    R = np.asarray(matrix, dtype=np.float64)[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw = math.degrees(math.atan2(-R[2, 0], sy))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return yaw, pitch, roll


def face_landmarks_from_array(
    all_pts: np.ndarray,
    width: int,
    height: int,
    pose: Optional[tuple[float, float, float]] = None,
    confidence: Optional[float] = None,
) -> FaceLandmarks:
    """
    Build the eye-bags FaceLandmarks (normalised coords) from a raw landmark
    array.

    Args:
        all_pts: (N, 2) or (N, 3) normalised landmark coords (x, y[, z]).
        width, height: source image size in pixels.
        pose: optional (yaw, pitch, roll) in degrees. When provided (from the
            facial transformation matrix) it OVERRIDES the legacy nose/eye-line
            heuristic, which is unreliable for the Tasks landmarker.
        confidence: optional face confidence override in [0, 1]. The Tasks
            landmarker only returns landmarks once its own presence/detection
            thresholds pass, so a high constant is appropriate; the z-std proxy
            below does not transfer from the legacy FaceMesh.
    """
    pts = np.asarray(all_pts, dtype=np.float32)
    n_lm = len(pts)
    xy = pts[:, :2]
    z = pts[:, 2] if pts.shape[1] > 2 else np.zeros(n_lm, dtype=np.float32)

    def _get(indices):
        valid = [i for i in indices if 0 <= i < n_lm]
        return xy[valid] if valid else None

    def _get1(idx):
        return xy[idx].copy() if 0 <= idx < n_lm else None

    # Pose roll from the eye-centre line (in-plane tilt).
    left_ctr = xy[[LEFT_EYE_OUTER, LEFT_EYE_INNER]].mean(axis=0)
    right_ctr = xy[[RIGHT_EYE_OUTER, RIGHT_EYE_INNER]].mean(axis=0)
    dx = right_ctr[0] - left_ctr[0]
    dy = right_ctr[1] - left_ctr[1]
    roll_deg = float(np.degrees(np.arctan2(dy, dx)))

    nose_tip = xy[1] if n_lm > 1 else np.array([0.5, 0.5], np.float32)
    eye_mid = (left_ctr + right_ctr) * 0.5
    interocular = float(np.hypot(dx, dy))

    if interocular > 1e-6:
        yaw_deg = float(np.clip((nose_tip[0] - eye_mid[0]) / interocular * 90.0, -60.0, 60.0))
        FRONTAL_NOSE_RATIO = 0.65
        nose_ratio = (nose_tip[1] - eye_mid[1]) / interocular
        pitch_deg = float(np.clip((FRONTAL_NOSE_RATIO - nose_ratio) / 0.25 * 45.0, -60.0, 60.0))
    else:
        yaw_deg = 0.0
        pitch_deg = 0.0

    conf = float(np.clip(1.0 - np.std(z) * 8.0, 0.1, 1.0))

    # Prefer the reliable matrix-derived pose / explicit confidence when given.
    if pose is not None:
        yaw_deg, pitch_deg, roll_deg = pose
    if confidence is not None:
        conf = float(confidence)

    return FaceLandmarks(
        success=True,
        height=height,
        width=width,
        left_outer=_get1(LEFT_EYE_OUTER),
        left_inner=_get1(LEFT_EYE_INNER),
        right_outer=_get1(RIGHT_EYE_OUTER),
        right_inner=_get1(RIGHT_EYE_INNER),
        left_lower_pts=_get(LEFT_LOWER_LID + LEFT_INFRAORBITAL),
        right_lower_pts=_get(RIGHT_LOWER_LID + RIGHT_INFRAORBITAL),
        left_cheek=_get(LEFT_CHEEK_REF),
        right_cheek=_get(RIGHT_CHEEK_REF),
        face_confidence=conf,
        pose_yaw_deg=yaw_deg,
        pose_pitch_deg=pitch_deg,
        pose_roll_deg=roll_deg,
    )


class SharedTasksLandmarker:
    """
    MediaPipe Tasks landmarker exposed through the eye-bags FaceLandmarker API.

    Args:
        model_path: path to the MediaPipe `face_landmarker.task` model. If None,
            falls back to the MEDIAPIPE_FACE_LANDMARKER_MODEL env var (handled by
            selfie_face_preprocess.PreprocessConfig).
        analyzer: inject a preconstructed analyzer/stub with `.analyze(rgb)`
            returning an object exposing `.landmarks` and `.face_count`
            (used by tests to avoid loading MediaPipe).
    """

    def __init__(self, model_path: Optional[str] = None, analyzer=None):
        self._analyzer = analyzer
        self._model_path = model_path
        # Cache of the most recent detection: (key, FaceLandmarks, all_pts_px, count)
        self._cache_key = None
        self._cache_lm: Optional[FaceLandmarks] = None
        self._cache_all_px: Optional[np.ndarray] = None
        self._cache_count: int = 0

    def _ensure_analyzer(self):
        if self._analyzer is None:
            from selfie_face_preprocess import PreprocessConfig
            from selfie_face_preprocess.mediapipe_adapter import MediaPipeFaceAnalyzer
            config = PreprocessConfig(face_landmarker_model_path=self._model_path) \
                if self._model_path else PreprocessConfig()
            self._analyzer = MediaPipeFaceAnalyzer(config)

    @staticmethod
    def _key(image) -> tuple:
        return (id(image), image.shape, str(image.dtype))

    def _run(self, image_bgr: np.ndarray):
        """Run the Tasks landmarker on a BGR image; populate the cache."""
        self._ensure_analyzer()
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])   # BGR -> RGB
        h, w = rgb.shape[:2]
        analysis = self._analyzer.analyze(rgb)
        count = int(getattr(analysis, "face_count", 0) or 0)

        if not getattr(analysis, "landmarks", None):
            self._cache_lm = FaceLandmarks(success=False, height=h, width=w,
                                           error_msg="No face detected")
            self._cache_all_px = None
            self._cache_count = count
            return

        all_norm = np.asarray(analysis.landmarks[0], dtype=np.float32)   # (N, 2|3)

        # Reliable pose from the facial transformation matrix when available.
        pose = None
        confidence = None
        matrices = getattr(analysis, "facial_transformation_matrixes", None)
        if matrices:
            try:
                pose = pose_from_transform_matrix(matrices[0])
                confidence = 1.0   # Tasks landmarker already gated on presence.
            except Exception:      # pragma: no cover - malformed matrix
                pose = None

        self._cache_lm = face_landmarks_from_array(
            all_norm, w, h, pose=pose, confidence=confidence)
        self._cache_all_px = all_norm[:, :2] * np.array([w, h], np.float32)
        self._cache_count = count

    def detect(self, image_bgr: np.ndarray) -> FaceLandmarks:
        """Drop-in replacement for eye-bags FaceLandmarker.detect (memoised)."""
        if image_bgr is None or image_bgr.ndim != 3:
            return FaceLandmarks(success=False, error_msg="Expected (H,W,3) array")
        key = self._key(image_bgr)
        if key != self._cache_key:
            self._run(image_bgr)
            self._cache_key = key
        return self._cache_lm

    def all_points_px(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Full landmark array in PIXEL coords for the given image (memoised)."""
        self.detect(image_bgr)
        return self._cache_all_px

    @property
    def face_count(self) -> int:
        return self._cache_count
