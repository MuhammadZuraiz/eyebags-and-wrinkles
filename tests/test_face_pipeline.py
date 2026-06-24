import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.face_landmarks import FaceLandmarks
from src.preprocessing.roi_cropper import CropResult, RoiPair
from src.face_analysis.pipeline import (
    FaceSkinAnalyzer, SCHEMA_VERSION, coral_grade, coral_confidence,
)
from src.wrinkles.infer import WrinkleResult
from src.wrinkles.regions import REGION_NAMES


# ── fakes ────────────────────────────────────────────────────────────────────

class FakeQuality:
    def __init__(self, accepted=True, reasons=None, retake_message="", face_detected=True):
        self.accepted = accepted
        self.reasons = reasons or []
        self.retake_message = retake_message
        self.details = {"face_detected": face_detected}


class FakeGate:
    def __init__(self, qr):
        self.qr = qr

    def check(self, image):
        return self.qr


class FakeLandmarker:
    def __init__(self, success=True, all_px=None):
        self.success = success
        self._all_px = all_px

    def detect(self, image):
        return FaceLandmarks(success=self.success, width=image.shape[1], height=image.shape[0])

    def all_points_px(self, image):
        return self._all_px


class FakeCropper:
    def __init__(self, ok=True):
        self.ok = ok

    def crop(self, image, landmarks):
        if not self.ok:
            f = CropResult(success=False, error_msg="roi failed")
            return RoiPair(left=f, right=f)
        crop = np.zeros((160, 256, 3), np.uint8)
        return RoiPair(left=CropResult(success=True, crop=crop),
                       right=CropResult(success=True, crop=crop))


class _Input:
    name = "image"


class FakeEyeBagSession:
    """Cycles through `sev_vectors` on successive run() calls (left, then right)."""
    def __init__(self, presence_logit=2.2, sev_vectors=None):
        self.presence_logit = presence_logit
        self.sev_vectors = sev_vectors or [[3.0, 1.0, -2.0, -3.0]]
        self._i = 0

    def get_inputs(self):
        return [_Input()]

    def run(self, _outputs, feeds):
        sev = self.sev_vectors[self._i % len(self.sev_vectors)]
        self._i += 1
        return (
            np.array([[self.presence_logit]], np.float32),
            np.array([sev], np.float32),
            np.array([0.0], np.float32),
        )


class FakeWrinkle:
    def analyze(self, rgb, landmarks_px=None, keep_mask=False):
        return WrinkleResult(
            overall_score=0.3, coverage_fraction=0.02,
            regions={n: 0.1 for n in REGION_NAMES},
            mask_available=True, detected=landmarks_px is not None,
        )


def _make_analyzer(gate, landmarker, eye_session=None, wrinkle=None, cropper=None):
    return FaceSkinAnalyzer(
        landmarker=landmarker,
        quality_gate=gate,
        eye_bag_session=eye_session,
        wrinkle_analyzer=wrinkle,
        cropper=cropper or FakeCropper(),
    )


IMG = np.zeros((400, 400, 3), np.uint8)


# ── CORAL numpy decode ───────────────────────────────────────────────────────

class CoralDecodeTests(unittest.TestCase):
    def test_grade_counting(self):
        self.assertEqual(coral_grade([5, 5, 5, 5]), 4)
        self.assertEqual(coral_grade([-5, -5, -5, -5]), 0)
        self.assertEqual(coral_grade([5, 1, -1, -5]), 2)

    def test_confidence_range(self):
        c = coral_confidence([5, 5, 5, 5])
        self.assertGreater(c, 0.9)
        self.assertLessEqual(c, 1.0)
        self.assertLess(coral_confidence([0, 0, 0, 0]), 0.05)


# ── pipeline contract + decisions ────────────────────────────────────────────

class PipelineTests(unittest.TestCase):
    def _assert_contract(self, r):
        for key in ("schema_version", "quality", "eye_bags", "wrinkles",
                    "confounders", "decision", "message", "disclaimer"):
            self.assertIn(key, r)
        self.assertEqual(r["schema_version"], SCHEMA_VERSION)
        self.assertIn(r["decision"],
                      {"show_guidance", "abstain", "retake_requested"})
        self.assertEqual(set(r["wrinkles"]["regions"].keys()), set(REGION_NAMES))
        json.dumps(r)   # must be JSON-serialisable

    def test_full_show_guidance(self):
        analyzer = _make_analyzer(
            FakeGate(FakeQuality()), FakeLandmarker(all_px=np.zeros((478, 2), np.float32)),
            eye_session=FakeEyeBagSession(sev_vectors=[[3, 1, -2, -3]]),
            wrinkle=FakeWrinkle(),
        )
        r = analyzer.analyze(IMG)
        self._assert_contract(r)
        self.assertEqual(r["decision"], "show_guidance")
        self.assertTrue(r["eye_bags"]["available"])
        self.assertIsNotNone(r["eye_bags"]["left"])
        self.assertIn("severity_label", r["eye_bags"]["left"])
        self.assertTrue(r["wrinkles"]["mask_available"])

    def test_retake_when_quality_rejected(self):
        analyzer = _make_analyzer(
            FakeGate(FakeQuality(accepted=False, reasons=["too_blurry"],
                                 retake_message="blurry")),
            FakeLandmarker(),
            eye_session=FakeEyeBagSession(), wrinkle=FakeWrinkle(),
        )
        r = analyzer.analyze(IMG)
        self._assert_contract(r)
        self.assertEqual(r["decision"], "retake_requested")
        self.assertEqual(r["message"], "blurry")
        self.assertFalse(r["eye_bags"]["available"])

    def test_abstain_when_no_landmarks(self):
        analyzer = _make_analyzer(
            FakeGate(FakeQuality()), FakeLandmarker(success=False),
            eye_session=FakeEyeBagSession(), wrinkle=FakeWrinkle(),
        )
        r = analyzer.analyze(IMG)
        self._assert_contract(r)
        self.assertEqual(r["decision"], "abstain")

    def test_asymmetry_abstain(self):
        analyzer = _make_analyzer(
            FakeGate(FakeQuality()), FakeLandmarker(all_px=np.zeros((478, 2), np.float32)),
            eye_session=FakeEyeBagSession(sev_vectors=[[5, 5, 5, 5], [-5, -5, -5, -5]]),
            wrinkle=FakeWrinkle(),
        )
        r = analyzer.analyze(IMG)
        self._assert_contract(r)
        self.assertEqual(r["decision"], "abstain")
        self.assertTrue(r["confounders"]["significant_asymmetry"])

    def test_wrinkle_only_when_no_eyebag_model(self):
        analyzer = _make_analyzer(
            FakeGate(FakeQuality()), FakeLandmarker(all_px=np.zeros((478, 2), np.float32)),
            eye_session=None, wrinkle=FakeWrinkle(),
        )
        r = analyzer.analyze(IMG)
        self._assert_contract(r)
        self.assertFalse(r["eye_bags"]["available"])
        self.assertTrue(r["wrinkles"]["mask_available"])
        self.assertEqual(r["decision"], "show_guidance")


if __name__ == "__main__":
    unittest.main()
