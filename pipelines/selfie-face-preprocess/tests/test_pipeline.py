from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selfie_face_preprocess.analysis import FaceAnalysis  # noqa: E402
from selfie_face_preprocess.config import PreprocessConfig  # noqa: E402
from selfie_face_preprocess.geometry import FACE_OVAL, LEFT_EYE, RIGHT_EYE  # noqa: E402
from selfie_face_preprocess.pipeline import preprocess_selfie  # noqa: E402


class FakeAnalyzer:
    def __init__(self, analysis: FaceAnalysis):
        self.analysis = analysis

    def analyze(self, rgb: np.ndarray) -> FaceAnalysis:
        return self.analysis


def fake_landmarks(width: int, height: int) -> list[list[float]]:
    points = np.zeros((478, 3), dtype=np.float32)
    points[:, 0] = 0.5
    points[:, 1] = 0.5
    points[:, 2] = 0.0

    for idx in LEFT_EYE:
        points[idx, :2] = [0.42, 0.38]
    for idx in RIGHT_EYE:
        points[idx, :2] = [0.58, 0.38]

    oval_xy = [
        (0.50, 0.20),
        (0.62, 0.23),
        (0.72, 0.34),
        (0.76, 0.52),
        (0.70, 0.72),
        (0.60, 0.84),
        (0.50, 0.88),
        (0.40, 0.84),
        (0.30, 0.72),
        (0.24, 0.52),
        (0.28, 0.34),
        (0.38, 0.23),
    ]
    for n, idx in enumerate(FACE_OVAL):
        points[idx, :2] = oval_xy[n % len(oval_xy)]
    points[1, :2] = [0.5, 0.52]
    points[152, :2] = [0.5, 0.88]
    points[234, :2] = [0.27, 0.52]
    points[454, :2] = [0.73, 0.52]
    return points.tolist()


def selfie_like_image(width: int = 320, height: int = 420, value: int = 125) -> np.ndarray:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    for y in range(height):
        image[y, :, 1] = np.clip(value + (y % 17), 0, 255)
    return image


class PipelineTests(unittest.TestCase):
    def test_pipeline_accepts_single_face(self) -> None:
        image = selfie_like_image()
        analysis = FaceAnalysis(face_count=1, landmarks=[fake_landmarks(320, 420)])
        config = PreprocessConfig(output_size=256, blur_threshold=0.0)
        result = preprocess_selfie(image, config, analyzer=FakeAnalyzer(analysis))

        self.assertTrue(result.accepted, result.reject_reasons)
        self.assertEqual(result.model_input.shape, (256, 256, 3))
        self.assertEqual(result.face_aligned.shape, (256, 256, 3))
        self.assertEqual(result.face_mask.shape, (256, 256))
        self.assertIn("aligned_after_enhance", result.quality)
        self.assertEqual(result.metadata["config"]["model_input_mode"], "aligned")

    def test_masked_fill_mode_changes_background(self) -> None:
        image = selfie_like_image()
        analysis = FaceAnalysis(face_count=1, landmarks=[fake_landmarks(320, 420)])
        config = PreprocessConfig(
            output_size=256,
            blur_threshold=0.0,
            model_input_mode="masked_fill",
        )
        result = preprocess_selfie(image, config, analyzer=FakeAnalyzer(analysis))

        self.assertTrue(result.accepted, result.reject_reasons)
        self.assertEqual(result.metadata["config"]["model_input_mode"], "masked_fill")
        self.assertIn("masked_fill_not_recommended_for_rgb_skin_models", result.warnings)
        self.assertFalse(np.array_equal(result.model_input[0, 0], result.face_aligned[0, 0]))

    def test_pipeline_rejects_no_face(self) -> None:
        image = selfie_like_image()
        result = preprocess_selfie(
            image,
            PreprocessConfig(output_size=128),
            analyzer=FakeAnalyzer(FaceAnalysis(face_count=0)),
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reject_reasons, ["no_face"])

    def test_pipeline_rejects_multiple_faces(self) -> None:
        image = selfie_like_image()
        result = preprocess_selfie(
            image,
            PreprocessConfig(output_size=128),
            analyzer=FakeAnalyzer(FaceAnalysis(face_count=2)),
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reject_reasons, ["multiple_faces"])

    def test_save_outputs(self) -> None:
        image = selfie_like_image()
        analysis = FaceAnalysis(face_count=1, landmarks=[fake_landmarks(320, 420)])
        result = preprocess_selfie(
            image,
            PreprocessConfig(output_size=128, blur_threshold=0.0),
            analyzer=FakeAnalyzer(analysis),
        )
        with tempfile.TemporaryDirectory() as tmp:
            written = result.save_outputs(tmp)
            self.assertTrue(written["metadata"].exists())
            self.assertTrue(written["model_input"].exists())


if __name__ == "__main__":
    unittest.main()
