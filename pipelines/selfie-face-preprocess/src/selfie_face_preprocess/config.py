from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_path(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


@dataclass(slots=True)
class PreprocessConfig:
    """Runtime settings for the selfie preprocessing pipeline."""

    output_size: int = 1024
    face_landmarker_model_path: str | Path | None = field(
        default_factory=lambda: _env_path("MEDIAPIPE_FACE_LANDMARKER_MODEL")
    )
    face_detector_model_path: str | Path | None = field(
        default_factory=lambda: _env_path("MEDIAPIPE_FACE_DETECTOR_MODEL")
    )

    model_input_mode: str = "aligned"
    max_faces_to_detect: int = 2
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5

    min_face_fraction: float = 0.18
    max_roll_degrees: float = 45.0
    max_yaw_offset_fraction: float = 0.24

    blur_threshold: float = 45.0
    min_luminance: float = 45.0
    max_luminance: float = 218.0
    max_glare_fraction: float = 0.03
    noise_threshold: float = 9.0

    enable_exposure: bool = True
    exposure_target_luma: float = 128.0
    exposure_gain_min: float = 0.80
    exposure_gain_max: float = 1.25

    enable_white_balance: bool = True
    white_balance_gain_min: float = 0.90
    white_balance_gain_max: float = 1.10

    enable_clahe: bool = True
    clahe_clip_limit: float = 1.4
    clahe_blend: float = 0.25

    enable_denoise: bool = True
    denoise_blend: float = 0.20
    denoise_h: float = 3.0

    mask_blur_radius: float = 2.0
    reject_on_quality: bool = True

    def __post_init__(self) -> None:
        if self.output_size <= 0:
            raise ValueError("output_size must be positive")
        if self.max_faces_to_detect < 1:
            raise ValueError("max_faces_to_detect must be at least 1")
        if self.model_input_mode not in {"aligned", "masked_fill"}:
            raise ValueError("model_input_mode must be 'aligned' or 'masked_fill'")
        for name in (
            "min_face_detection_confidence",
            "min_face_presence_confidence",
            "min_face_fraction",
            "max_glare_fraction",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
