from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .analysis import FaceAnalysis
from .config import PreprocessConfig
from .enhance import enhance_for_skin
from .geometry import (
    apply_affine_to_points,
    build_face_mask,
    estimate_alignment,
    estimate_pose,
    face_bbox,
    fill_outside_face,
    normalized_to_pixels,
    warp_affine_rgb,
)
from .image_io import load_rgb_image
from .mediapipe_adapter import MediaPipeFaceAnalyzer
from .quality import quality_reject_reasons, score_image
from .result import FacePreprocessResult


class FaceAnalyzer(Protocol):
    def analyze(self, rgb: np.ndarray) -> FaceAnalysis:
        ...


def preprocess_selfie(
    input_image: str | Path | Image.Image | np.ndarray,
    config: PreprocessConfig | None = None,
    *,
    analyzer: FaceAnalyzer | None = None,
) -> FacePreprocessResult:
    """Convert a selfie into a normalized face-only model input."""

    config = config or PreprocessConfig()
    rgb, source = load_rgb_image(input_image)
    height, width = rgb.shape[:2]
    input_quality = score_image(rgb)

    warnings: list[str] = []
    reject_reasons: list[str] = []
    metadata: dict[str, object] = {
        "input": {
            **source,
            "width": width,
            "height": height,
        },
        "config": _config_metadata(config),
    }

    face_analyzer = analyzer or MediaPipeFaceAnalyzer(config)
    analysis = face_analyzer.analyze(rgb)
    metadata["analysis"] = analysis.to_metadata()

    if analysis.face_count == 0:
        return _rejected("no_face", reject_reasons, warnings, input_quality, metadata)
    if analysis.face_count > 1:
        return _rejected("multiple_faces", reject_reasons, warnings, input_quality, metadata)
    if not analysis.landmarks:
        return _rejected("landmarks_failed", reject_reasons, warnings, input_quality, metadata)

    try:
        points = normalized_to_pixels(analysis.landmarks[0], width, height)
        bbox = face_bbox(points)
        face_fraction = max(bbox["width"], bbox["height"]) / float(min(width, height))
        pose = estimate_pose(points)
        matrix, transform = estimate_alignment(points, config.output_size)
    except (IndexError, ValueError) as exc:
        metadata["landmark_error"] = str(exc)
        return _rejected("landmarks_failed", reject_reasons, warnings, input_quality, metadata)

    if face_fraction < config.min_face_fraction:
        reject_reasons.append("face_too_small")

    if (
        abs(transform["roll_degrees"]) > config.max_roll_degrees
        or abs(pose["yaw_offset_fraction"]) > config.max_yaw_offset_fraction
    ):
        reject_reasons.append("extreme_pose")

    aligned = warp_affine_rgb(rgb, matrix, config.output_size)
    transformed_points = apply_affine_to_points(points, matrix)
    mask, soft_mask = build_face_mask(
        transformed_points,
        config.output_size,
        blur_radius=config.mask_blur_radius,
    )
    aligned_quality = score_image(aligned, mask)

    quality_reasons = quality_reject_reasons(aligned_quality, config)
    if config.reject_on_quality:
        reject_reasons.extend(quality_reasons)
    else:
        warnings.extend(quality_reasons)

    enhanced, enhancement_params, enhancement_warnings = enhance_for_skin(aligned, config, mask)
    warnings.extend(enhancement_warnings)
    enhanced_quality = score_image(enhanced, mask)
    if config.model_input_mode == "aligned":
        model_input = enhanced
    else:
        warnings.append("masked_fill_not_recommended_for_rgb_skin_models")
        model_input = fill_outside_face(enhanced, mask, soft_mask)

    metadata.update(
        {
            "face": {
                "bbox": bbox,
                "face_fraction": float(face_fraction),
                "pose": pose,
                "landmarks_normalized": analysis.landmarks[0],
                "landmarks_aligned": transformed_points.tolist(),
                "facial_transformation_matrix": (
                    analysis.facial_transformation_matrixes[0]
                    if analysis.facial_transformation_matrixes
                    else None
                ),
            },
            "transform": {
                **transform,
                "affine_2x3": matrix.tolist(),
            },
            "preprocessing": enhancement_params,
        }
    )

    quality = {
        "input": input_quality,
        "aligned_before_enhance": aligned_quality,
        "aligned_after_enhance": enhanced_quality,
    }
    return FacePreprocessResult(
        accepted=not reject_reasons,
        reject_reasons=_dedupe(reject_reasons),
        warnings=_dedupe(warnings),
        quality=quality,
        metadata=metadata,
        model_input=model_input,
        face_aligned=aligned,
        face_mask=mask,
    )


def _rejected(
    reason: str,
    reject_reasons: list[str],
    warnings: list[str],
    input_quality: dict[str, object],
    metadata: dict[str, object],
) -> FacePreprocessResult:
    reject_reasons.append(reason)
    return FacePreprocessResult(
        accepted=False,
        reject_reasons=_dedupe(reject_reasons),
        warnings=_dedupe(warnings),
        quality={"input": input_quality},
        metadata=metadata,
    )


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _config_metadata(config: PreprocessConfig) -> dict[str, object]:
    return {
        "output_size": config.output_size,
        "max_faces_to_detect": config.max_faces_to_detect,
        "min_face_detection_confidence": config.min_face_detection_confidence,
        "min_face_presence_confidence": config.min_face_presence_confidence,
        "min_face_fraction": config.min_face_fraction,
        "max_roll_degrees": config.max_roll_degrees,
        "max_yaw_offset_fraction": config.max_yaw_offset_fraction,
        "blur_threshold": config.blur_threshold,
        "min_luminance": config.min_luminance,
        "max_luminance": config.max_luminance,
        "max_glare_fraction": config.max_glare_fraction,
        "noise_threshold": config.noise_threshold,
        "reject_on_quality": config.reject_on_quality,
        "model_input_mode": config.model_input_mode,
        "face_landmarker_model_path": str(config.face_landmarker_model_path)
        if config.face_landmarker_model_path
        else None,
        "face_detector_model_path": str(config.face_detector_model_path)
        if config.face_detector_model_path
        else None,
    }
