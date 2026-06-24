from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selfie_face_preprocess.geometry import (  # noqa: E402
    FACE_OVAL,
    LEFT_EYE,
    RIGHT_EYE,
    apply_affine_to_points,
    build_face_mask,
    estimate_alignment,
    fill_outside_face,
)


def fake_points() -> np.ndarray:
    points = np.zeros((478, 2), dtype=np.float32)
    points[:] = [160, 160]
    for idx in LEFT_EYE:
        points[idx] = [120, 130]
    for idx in RIGHT_EYE:
        points[idx] = [200, 132]
    oval_xy = [
        (160, 70),
        (190, 76),
        (220, 104),
        (236, 150),
        (220, 218),
        (190, 250),
        (160, 260),
        (130, 250),
        (100, 218),
        (84, 150),
        (100, 104),
        (130, 76),
    ]
    for n, idx in enumerate(FACE_OVAL):
        points[idx] = oval_xy[n % len(oval_xy)]
    points[1] = [160, 165]
    points[152] = [160, 260]
    points[234] = [92, 165]
    points[454] = [228, 165]
    return points


class GeometryTests(unittest.TestCase):
    def test_alignment_levels_eyes(self) -> None:
        points = fake_points()
        matrix, details = estimate_alignment(points, output_size=256)
        transformed = apply_affine_to_points(points, matrix)
        left_y = transformed[LEFT_EYE, 1].mean()
        right_y = transformed[RIGHT_EYE, 1].mean()
        self.assertAlmostEqual(left_y, right_y, delta=0.01)
        self.assertAlmostEqual(details["target_eye_distance_px"], 87.04, places=2)

    def test_mask_and_fill(self) -> None:
        points = fake_points()
        matrix, _ = estimate_alignment(points, output_size=256)
        transformed = apply_affine_to_points(points, matrix)
        mask, soft = build_face_mask(transformed, 256, blur_radius=1.0)
        self.assertEqual(mask.shape, (256, 256))
        self.assertGreater(mask.mean(), 10)
        image = np.full((256, 256, 3), [120, 90, 70], dtype=np.uint8)
        image[mask == 0] = [5, 5, 5]
        filled = fill_outside_face(image, mask, soft)
        self.assertEqual(filled.shape, image.shape)
        self.assertGreater(filled[0, 0, 0], 50)


if __name__ == "__main__":
    unittest.main()
