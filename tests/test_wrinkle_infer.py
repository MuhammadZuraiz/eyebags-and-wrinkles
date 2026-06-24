import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.wrinkles.infer import (
    WrinkleAnalyzer, logits_to_mask, normalize_unet_input,
)
from src.wrinkles.preprocess import OUTPUT_SIZE
from src.wrinkles.regions import REGION_NAMES


def _synthetic_landmarks_px(size=900, n=478, seed=1):
    """Plausible in-frame landmark cloud so region hulls are non-degenerate."""
    rng = np.random.default_rng(seed)
    return rng.uniform(size * 0.2, size * 0.8, size=(n, 2)).astype(np.float32)


class _StubUNetSession:
    """Returns fixed (1,2,S,S) logits with a central 'wrinkle' block (class 1)."""
    def __init__(self, size=OUTPUT_SIZE):
        self.size = size

    class _In:
        name = "input"

    def get_inputs(self):
        return [self._In()]

    def run(self, _outputs, feeds):
        s = self.size
        logits = np.zeros((1, 2, s, s), np.float32)
        logits[0, 0] = 1.0                       # class 0 baseline
        logits[0, 1, s // 4:3 * s // 4, s // 4:3 * s // 4] = 5.0   # class 1 block
        return [logits]


class NormalizeTests(unittest.TestCase):
    def test_input_assembly(self):
        rgb = np.full((OUTPUT_SIZE, OUTPUT_SIZE, 3), 255, np.uint8)
        tex = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), np.uint8)
        x = normalize_unet_input(rgb, tex)
        self.assertEqual(x.shape, (1, 4, OUTPUT_SIZE, OUTPUT_SIZE))
        self.assertAlmostEqual(float(x[0, 0].max()), 1.0, places=5)    # white -> +1
        self.assertAlmostEqual(float(x[0, 3].min()), -1.0, places=5)   # black tex -> -1

    def test_logits_to_mask(self):
        logits = np.zeros((1, 2, 8, 8), np.float32)
        logits[0, 1, 2:6, 2:6] = 9.0
        mask = logits_to_mask(logits)
        self.assertEqual(mask.shape, (8, 8))
        self.assertEqual(set(np.unique(mask)).issubset({0, 255}), True)
        self.assertEqual(int(mask[4, 4]), 255)
        self.assertEqual(int(mask[0, 0]), 0)


class WrinkleAnalyzerTests(unittest.TestCase):
    def test_analyze_with_landmarks(self):
        analyzer = WrinkleAnalyzer(session=_StubUNetSession())
        rgb = np.full((900, 900, 3), 160, np.uint8)
        lm = _synthetic_landmarks_px(900)
        result = analyzer.analyze(rgb, landmarks_px=lm, keep_mask=True)

        self.assertTrue(result.mask_available)
        self.assertTrue(result.detected)              # landmark-driven crop
        self.assertEqual(result.mask.shape, (OUTPUT_SIZE, OUTPUT_SIZE))
        self.assertGreater(result.coverage_fraction, 0.0)
        self.assertGreaterEqual(result.overall_score, 0.0)
        self.assertLessEqual(result.overall_score, 1.0)
        self.assertEqual(set(result.regions.keys()), set(REGION_NAMES))
        d = result.to_dict()
        self.assertIn("overall_score", d)
        self.assertIn("regions", d)

    def test_analyze_without_landmarks_uses_fallback(self):
        analyzer = WrinkleAnalyzer(session=_StubUNetSession())
        rgb = np.full((900, 900, 3), 160, np.uint8)
        result = analyzer.analyze(rgb, landmarks_px=None)
        self.assertTrue(result.mask_available)
        self.assertFalse(result.detected)             # ellipse fallback
        # No landmarks -> region scores all zero, but coverage still computed.
        self.assertEqual(set(result.regions.values()), {0.0})


if __name__ == "__main__":
    unittest.main()
