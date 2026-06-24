from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [263, 362, 386, 374]
NOSE_TIP = 1
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
FACE_OVAL = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]


def normalized_to_pixels(landmarks: list[list[float]], width: int, height: int) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("landmarks must be a list of [x, y, z] points")
    pixels = points[:, :2].copy()
    pixels[:, 0] *= float(width)
    pixels[:, 1] *= float(height)
    return pixels


def face_bbox(points: np.ndarray) -> dict[str, float]:
    oval = _points_for_indices(points, FACE_OVAL)
    min_xy = np.min(oval, axis=0)
    max_xy = np.max(oval, axis=0)
    return {
        "x": float(min_xy[0]),
        "y": float(min_xy[1]),
        "width": float(max_xy[0] - min_xy[0]),
        "height": float(max_xy[1] - min_xy[1]),
    }


def estimate_alignment(points: np.ndarray, output_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    eye_a = _mean_point(points, LEFT_EYE)
    eye_b = _mean_point(points, RIGHT_EYE)
    left_eye, right_eye = sorted([eye_a, eye_b], key=lambda p: p[0])
    eye_mid = (left_eye + right_eye) / 2.0
    eye_delta = right_eye - left_eye
    eye_distance = float(np.linalg.norm(eye_delta))
    if eye_distance <= 1.0:
        raise ValueError("eye landmarks are too close to estimate alignment")

    roll = math.degrees(math.atan2(float(eye_delta[1]), float(eye_delta[0])))
    target_eye_distance = float(output_size) * 0.34
    scale = target_eye_distance / eye_distance
    target_mid = np.array([output_size * 0.5, output_size * 0.38], dtype=np.float32)

    matrix = _rotation_scale_matrix(eye_mid, -roll, scale)
    mapped_mid = apply_affine_to_points(np.asarray([eye_mid], dtype=np.float32), matrix)[0]
    matrix[:, 2] += target_mid - mapped_mid

    details = {
        "roll_degrees": float(roll),
        "scale": float(scale),
        "eye_distance_px": eye_distance,
        "target_eye_distance_px": target_eye_distance,
        "eye_midpoint_px": eye_mid.tolist(),
        "target_eye_midpoint_px": target_mid.tolist(),
    }
    return matrix, details


def estimate_pose(points: np.ndarray) -> dict[str, float]:
    eye_mid = (_mean_point(points, LEFT_EYE) + _mean_point(points, RIGHT_EYE)) / 2.0
    nose = points[NOSE_TIP]
    left_cheek = points[LEFT_CHEEK]
    right_cheek = points[RIGHT_CHEEK]
    cheek_width = max(float(abs(right_cheek[0] - left_cheek[0])), 1.0)
    yaw_offset = float((nose[0] - eye_mid[0]) / cheek_width)
    chin = points[CHIN]
    face_height = max(float(abs(chin[1] - eye_mid[1])), 1.0)
    nose_drop = float((nose[1] - eye_mid[1]) / face_height)
    return {
        "yaw_offset_fraction": yaw_offset,
        "nose_drop_fraction": nose_drop,
    }


def apply_affine_to_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    hom = np.concatenate([points.astype(np.float32), ones], axis=1)
    return hom @ matrix.T


def warp_affine_rgb(rgb: np.ndarray, matrix: np.ndarray, output_size: int) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError:
        image = Image.fromarray(rgb)
        inv = _invert_affine(matrix)
        coeffs = tuple(float(x) for x in inv.reshape(-1))
        warped = image.transform(
            (output_size, output_size),
            Image.Transform.AFFINE,
            coeffs,
            resample=Image.Resampling.BICUBIC,
            fillcolor=tuple(int(x) for x in np.median(rgb.reshape(-1, 3), axis=0)),
        )
        return np.asarray(warped, dtype=np.uint8)

    return cv2.warpAffine(
        rgb,
        matrix.astype(np.float32),
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def build_face_mask(
    transformed_points: np.ndarray,
    output_size: int,
    blur_radius: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    oval = _points_for_indices(transformed_points, FACE_OVAL)
    mask = Image.new("L", (output_size, output_size), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon([tuple(map(float, point)) for point in oval], fill=255)

    if blur_radius > 0:
        soft = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    else:
        soft = mask
    soft_arr = np.asarray(soft, dtype=np.float32) / 255.0
    binary = (soft_arr >= 0.5).astype(np.uint8) * 255
    return binary, soft_arr


def fill_outside_face(rgb: np.ndarray, mask: np.ndarray, soft_mask: np.ndarray | None = None) -> np.ndarray:
    mask_bool = np.asarray(mask) > 0
    if mask_bool.shape != rgb.shape[:2] or not mask_bool.any():
        median = np.median(rgb.reshape(-1, 3), axis=0)
        return np.full_like(rgb, median.astype(np.uint8))

    median = np.median(rgb[mask_bool], axis=0).astype(np.float32)
    alpha = soft_mask if soft_mask is not None and soft_mask.shape == mask_bool.shape else mask_bool.astype(np.float32)
    alpha = np.clip(alpha[..., None], 0.0, 1.0)
    filled = rgb.astype(np.float32) * alpha + median.reshape(1, 1, 3) * (1.0 - alpha)
    return np.clip(filled, 0, 255).astype(np.uint8)


def _mean_point(points: np.ndarray, indices: list[int]) -> np.ndarray:
    return np.mean(_points_for_indices(points, indices), axis=0)


def _points_for_indices(points: np.ndarray, indices: list[int]) -> np.ndarray:
    max_index = max(indices)
    if len(points) <= max_index:
        raise ValueError(f"expected at least {max_index + 1} face landmarks")
    return points[indices].astype(np.float32)


def _rotation_scale_matrix(center: np.ndarray, angle_degrees: float, scale: float) -> np.ndarray:
    theta = math.radians(angle_degrees)
    alpha = scale * math.cos(theta)
    beta = scale * math.sin(theta)
    cx, cy = float(center[0]), float(center[1])
    return np.asarray(
        [
            [alpha, -beta, (1.0 - alpha) * cx + beta * cy],
            [beta, alpha, -beta * cx + (1.0 - alpha) * cy],
        ],
        dtype=np.float32,
    )


def _invert_affine(matrix: np.ndarray) -> np.ndarray:
    full = np.vstack([matrix, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
    inv = np.linalg.inv(full)
    return inv[:2, :]
