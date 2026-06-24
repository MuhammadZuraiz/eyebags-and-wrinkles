"""Verify the inference output matches the spec's output contract (Section 2)."""
import pytest


REQUIRED_TOP_KEYS  = {"eye_bags", "confounders", "quality", "decision"}
REQUIRED_EYE_KEYS  = {"present_probability", "severity_grade", "severity_label", "confidence"}
VALID_DECISIONS    = {"show_guidance", "retake_requested", "abstain"}
VALID_LABELS       = {"not_present", "mild", "moderate", "pronounced", "very_pronounced"}


def validate_contract(result: dict):
    assert REQUIRED_TOP_KEYS <= set(result.keys()), f"Missing keys: {REQUIRED_TOP_KEYS - set(result)}"
    assert result["decision"] in VALID_DECISIONS

    for side in ("left", "right"):
        eye = result["eye_bags"][side]
        if eye is not None:   # None allowed on retake/abstain
            assert REQUIRED_EYE_KEYS <= set(eye.keys())
            assert 0.0 <= eye["present_probability"] <= 1.0
            assert eye["severity_grade"] in {0, 1, 2, 3, 4}
            assert eye["severity_label"] in VALID_LABELS
            assert 0.0 <= eye["confidence"] <= 1.0


def test_static_response_shapes():
    from src.deployment.inference import DermaLensPipeline
    # Use the response assembler directly (no model needed)
    resp = DermaLensPipeline._response(
        decision="show_guidance",
        quality={"accepted": True, "pose_ok": True, "lighting_ok": True,
                 "blur_ok": True, "face_detected": True},
        confounders={"dark_circles_visible": True,
                     "makeup_detected_or_suspected": False,
                     "significant_asymmetry": False},
        left={"present_probability": 0.87, "severity_grade": 2,
              "severity_label": "moderate", "confidence": 0.81},
        right={"present_probability": 0.79, "severity_grade": 2,
               "severity_label": "moderate", "confidence": 0.76},
    )
    validate_contract(resp)


def test_retake_response():
    from src.deployment.inference import DermaLensPipeline
    resp = DermaLensPipeline._response(
        decision="retake_requested",
        quality={"accepted": False, "pose_ok": False, "lighting_ok": True,
                 "blur_ok": True, "face_detected": True},
    )
    validate_contract(resp)
    assert resp["eye_bags"]["left"] is None
