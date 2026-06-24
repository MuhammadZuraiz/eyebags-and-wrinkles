from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selfie_face_preprocess.config import PreprocessConfig  # noqa: E402
from selfie_face_preprocess.quality import quality_reject_reasons, score_image  # noqa: E402


class QualityTests(unittest.TestCase):
    def test_dark_image_rejects(self) -> None:
        image = np.full((128, 128, 3), 20, dtype=np.uint8)
        quality = score_image(image)
        self.assertIn("too_dark", quality_reject_reasons(quality, PreprocessConfig()))

    def test_bright_image_rejects(self) -> None:
        image = np.full((128, 128, 3), 245, dtype=np.uint8)
        quality = score_image(image)
        reasons = quality_reject_reasons(quality, PreprocessConfig())
        self.assertIn("too_bright", reasons)
        self.assertIn("heavy_glare", reasons)

    def test_checkerboard_is_not_blurry(self) -> None:
        tile = np.indices((128, 128)).sum(axis=0) % 2
        image = np.repeat((tile * 255).astype(np.uint8)[..., None], 3, axis=2)
        quality = score_image(image)
        self.assertGreater(quality["blur_laplacian_var"], 45.0)


if __name__ == "__main__":
    unittest.main()
