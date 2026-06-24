import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.face_analysis.visualize import render_overlays, encode_png_b64


def _eye_res(grade=1):
    return {"present_probability": 0.6, "severity_grade": grade,
            "severity_label": "mild", "confidence": 0.8}


def _debug_bundle(size=160):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    crop = rng.integers(0, 255, (160, 256, 3), dtype=np.uint8)
    cr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), np.uint8); mask[40:120, 40:120] = 255
    fmask = np.full((size, size), 255, np.uint8)
    lm = rng.uniform(size * 0.2, size * 0.8, (478, 2)).astype(np.float32)
    return {
        "image_bgr": img[:, :, ::-1].copy(),
        "eye_bags": {
            "left_bbox": (10, 10, 80, 60), "right_bbox": (90, 10, 150, 60),
            "left_crop": crop, "right_crop": crop,
            "left": _eye_res(0), "right": _eye_res(2),
        },
        "wrinkles": {
            "mask": mask, "crop_rgb": cr, "masked_rgb": cr,
            "texture": (mask // 2).astype(np.uint8), "face_mask": fmask,
            "landmarks_crop": lm, "regions": {"forehead": 0.1},
        },
    }


class VisualizeTests(unittest.TestCase):
    def test_encode_png_b64(self):
        uri = encode_png_b64(np.zeros((20, 30, 3), np.uint8))
        self.assertTrue(uri.startswith("data:image/png;base64,"))

    def test_render_all_overlays(self):
        ov = render_overlays(_debug_bundle())
        for key in ("eye_bag_overlay", "eye_crops", "wrinkle_overlay",
                    "wrinkle_regions", "texture"):
            self.assertIn(key, ov)
            self.assertTrue(ov[key].startswith("data:image/png;base64,"))

    def test_empty_debug(self):
        self.assertEqual(render_overlays(None), {})
        self.assertEqual(render_overlays({}), {})


if __name__ == "__main__":
    unittest.main()
