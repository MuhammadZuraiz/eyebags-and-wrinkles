import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.wrinkles.texture import generate_texture_map, generate_texture_map_from_masked_face


def _synthetic_face(size=256):
    rng = np.random.default_rng(0)
    base = np.full((size, size, 3), 150, np.uint8)
    # Add some high-frequency "wrinkle-like" texture.
    noise = rng.integers(-25, 25, size=(size, size, 1), endpoint=True)
    return np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)


class TextureMapTests(unittest.TestCase):
    def test_shape_and_dtype(self):
        face = _synthetic_face()
        tex = generate_texture_map(face)
        self.assertEqual(tex.shape, (256, 256))
        self.assertEqual(tex.dtype, np.uint8)
        self.assertGreaterEqual(int(tex.min()), 0)
        self.assertLessEqual(int(tex.max()), 255)

    def test_deterministic(self):
        face = _synthetic_face()
        a = generate_texture_map(face)
        b = generate_texture_map(face)
        self.assertTrue(np.array_equal(a, b))

    def test_mask_zeroes_outside_face(self):
        face = _synthetic_face()
        mask = np.zeros((256, 256), np.uint8)
        mask[64:192, 64:192] = 255
        tex = generate_texture_map_from_masked_face(face, face_mask=mask)
        # Everything outside the mask must be exactly zero.
        self.assertEqual(int(tex[mask == 0].max()), 0)

    def test_alias_matches(self):
        face = _synthetic_face()
        self.assertTrue(np.array_equal(
            generate_texture_map(face), generate_texture_map_from_masked_face(face)))


if __name__ == "__main__":
    unittest.main()
