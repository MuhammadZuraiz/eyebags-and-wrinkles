from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .analysis import FaceAnalysis, FaceBox
from .config import PreprocessConfig


class MediaPipeConfigurationError(RuntimeError):
    """Raised when MediaPipe or required task models are not available."""


class MediaPipeFaceAnalyzer:
    """Small adapter around MediaPipe Tasks for still-image face analysis."""

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.landmarker_model_path = _required_model_path(
            config.face_landmarker_model_path,
            "face_landmarker_model_path",
        )
        self.detector_model_path = _optional_model_path(config.face_detector_model_path)

    def analyze(self, rgb: np.ndarray) -> FaceAnalysis:
        mp = _import_mediapipe()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        detector_boxes = self._detect_boxes(mp, mp_image) if self.detector_model_path else []
        landmarks, matrices = self._detect_landmarks(mp, mp_image)
        face_count = max(len(detector_boxes), len(landmarks))

        return FaceAnalysis(
            face_count=face_count,
            landmarks=landmarks,
            detector_boxes=detector_boxes,
            facial_transformation_matrixes=matrices,
        )

    def _detect_boxes(self, mp: Any, mp_image: Any) -> list[FaceBox]:
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(self.detector_model_path)),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=self.config.min_face_detection_confidence,
        )
        with FaceDetector.create_from_options(options) as detector:
            result = detector.detect(mp_image)

        boxes: list[FaceBox] = []
        for detection in getattr(result, "detections", []) or []:
            bbox = detection.bounding_box
            score = None
            categories = getattr(detection, "categories", None)
            if categories:
                score = float(categories[0].score)
            boxes.append(
                FaceBox(
                    origin_x=int(bbox.origin_x),
                    origin_y=int(bbox.origin_y),
                    width=int(bbox.width),
                    height=int(bbox.height),
                    score=score,
                )
            )
        return boxes

    def _detect_landmarks(self, mp: Any, mp_image: Any) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.landmarker_model_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=self.config.max_faces_to_detect,
            min_face_detection_confidence=self.config.min_face_detection_confidence,
            min_face_presence_confidence=self.config.min_face_presence_confidence,
            output_facial_transformation_matrixes=True,
        )
        with FaceLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(mp_image)

        landmarks: list[list[list[float]]] = []
        for face in getattr(result, "face_landmarks", []) or []:
            landmarks.append([[float(lm.x), float(lm.y), float(lm.z)] for lm in face])

        matrices: list[list[list[float]]] = []
        for matrix in getattr(result, "facial_transformation_matrixes", []) or []:
            arr = np.asarray(matrix, dtype=np.float32)
            if arr.size == 16:
                arr = arr.reshape(4, 4)
            matrices.append(arr.tolist())

        return landmarks, matrices


def _required_model_path(path: str | Path | None, field_name: str) -> Path:
    if path is None:
        raise MediaPipeConfigurationError(
            f"{field_name} is required. Pass --face-landmarker-model or set "
            "MEDIAPIPE_FACE_LANDMARKER_MODEL."
        )
    resolved = Path(path)
    if not resolved.exists():
        raise MediaPipeConfigurationError(f"{field_name} does not exist: {resolved}")
    return resolved


def _optional_model_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        raise MediaPipeConfigurationError(f"face_detector_model_path does not exist: {resolved}")
    return resolved


def _import_mediapipe() -> Any:
    try:
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        raise MediaPipeConfigurationError(
            "mediapipe is not installed. Install the full extras with "
            'python -m pip install -e ".[full]" in a Python 3.10-3.12 environment.'
        ) from exc
    return mp
