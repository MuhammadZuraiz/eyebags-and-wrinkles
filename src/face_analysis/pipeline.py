#!/usr/bin/env python3
"""
Unified face-skin analysis orchestrator (onnxruntime, torch-free).

    selfie (BGR)
      -> QualityGate (shared landmarker, one MediaPipe pass)
      -> SharedTasksLandmarker.detect  (cached)
      ├─ eye-bags: RoiCropper -> eye_bag.onnx (per eye) -> presence/severity
      └─ wrinkles: WrinkleAnalyzer -> wrinkle_unet.onnx -> coverage/regions
      -> unified JSON contract

All heavy deps (onnxruntime, mediapipe, selfie_face_preprocess) are imported
lazily or injected, so this module imports cleanly for unit tests. The eye-bag
CORAL decode and abstention rules are ported to numpy from
`src.models.ordinal_head` / `src.deployment.inference` (which require torch).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from src.preprocessing.roi_cropper import RoiCropper
from src.wrinkles.infer import empty_wrinkle_result

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

# ImageNet normalisation — must match src/data/augmentations.get_val_transforms.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
ROI_W, ROI_H = 256, 160

SEVERITY_LABELS = {
    0: "not_present", 1: "mild", 2: "moderate", 3: "pronounced", 4: "very_pronounced",
}

# Spec Section 6 thresholds (ported from src/deployment/inference.py)
MIN_MEAN_CONFIDENCE = 0.40
ASYMMETRY_GRADE_DIFF = 2

MSG_ASYMMETRY = (
    "The visible appearance is uneven between the two eye areas.\n"
    "DermaLens cannot determine the cause.\n"
    "Consider seeking professional advice if this is new, persistent, painful, or concerning."
)
MSG_POOR_QUALITY = (
    "We could not analyse the under-eye area reliably.\n"
    "Please retake the photo in even, natural lighting with your eyes fully open."
)
MSG_FOOTER = "This is cosmetic skincare guidance, not a medical diagnosis."
DISCLAIMER = MSG_FOOTER


# ── numpy CORAL decode (ports src.models.ordinal_head) ──────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float32)))


def coral_grade(sev_logits: np.ndarray) -> int:
    return int((_sigmoid(sev_logits) > 0.5).sum())


def coral_confidence(sev_logits: np.ndarray) -> float:
    p = _sigmoid(sev_logits)
    return float(np.mean(2.0 * np.abs(p - 0.5)))


def preprocess_eye_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """(160,256,3) BGR crop -> (1,3,160,256) ImageNet-normalised float32 RGB."""
    rgb = crop_bgr[:, :, ::-1].astype(np.float32) / 255.0     # BGR->RGB, [0,1]
    if rgb.shape[:2] != (ROI_H, ROI_W):
        import cv2
        rgb = cv2.resize(rgb, (ROI_W, ROI_H), interpolation=cv2.INTER_LINEAR)
    norm = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(norm, (2, 0, 1))
    return chw[None, ...].astype(np.float32)


class FaceSkinAnalyzer:
    """
    Unified analyzer. Construct with `.from_paths(...)` for real ONNX models, or
    inject components directly (tests).

    Injected components:
        landmarker:       object with .detect(bgr)->FaceLandmarks and
                          .all_points_px(bgr)->(N,2)|None  (SharedTasksLandmarker).
        quality_gate:     object with .check(bgr)->QualityResult.
        eye_bag_session:  onnxruntime session (or stub) for the eye-bag model.
        wrinkle_analyzer: src.wrinkles.infer.WrinkleAnalyzer (or stub).
    """

    def __init__(
        self,
        *,
        landmarker=None,
        quality_gate=None,
        eye_bag_session=None,
        wrinkle_analyzer=None,
        eye_bag_input_name: Optional[str] = None,
        cropper: Optional[RoiCropper] = None,
    ):
        self.landmarker = landmarker
        self.quality_gate = quality_gate
        self.eye_bag_session = eye_bag_session
        self.wrinkle_analyzer = wrinkle_analyzer
        self.cropper = cropper or RoiCropper()
        if eye_bag_session is not None and eye_bag_input_name is None:
            eye_bag_input_name = eye_bag_session.get_inputs()[0].name
        self.eye_bag_input_name = eye_bag_input_name

    # ── construction from model files ──────────────────────────────────────
    @classmethod
    def from_paths(
        cls,
        eye_bag_onnx: Optional[str] = None,
        wrinkle_onnx: Optional[str] = None,
        landmarker_model: Optional[str] = None,
        providers=None,
    ) -> "FaceSkinAnalyzer":
        from src.preprocessing.quality_gate import QualityGate
        from src.wrinkles.infer import WrinkleAnalyzer
        from .landmarks import SharedTasksLandmarker

        landmarker = SharedTasksLandmarker(model_path=landmarker_model)
        quality_gate = QualityGate(landmarker=landmarker)

        eye_bag_session = None
        if eye_bag_onnx:
            import onnxruntime as ort
            eye_bag_session = ort.InferenceSession(
                str(eye_bag_onnx), providers=providers or ["CPUExecutionProvider"])

        wrinkle_analyzer = (
            WrinkleAnalyzer(str(wrinkle_onnx), providers=providers) if wrinkle_onnx else None
        )
        return cls(
            landmarker=landmarker,
            quality_gate=quality_gate,
            eye_bag_session=eye_bag_session,
            wrinkle_analyzer=wrinkle_analyzer,
        )

    # ── main entry ─────────────────────────────────────────────────────────
    def analyze(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        result, _ = self._analyze(image_bgr, collect_debug=False)
        return result

    def analyze_debug(self, image_bgr: np.ndarray):
        """Like analyze() but also returns a debug dict of numpy intermediates
        (ROI bboxes/crops, wrinkle mask, masked/unmasked crop, texture, face mask,
        landmarks-in-crop) for overlay rendering. Returns (result, debug)."""
        return self._analyze(image_bgr, collect_debug=True)

    def _analyze(self, image_bgr: np.ndarray, collect_debug: bool = False):
        debug: Optional[Dict[str, Any]] = {"image_bgr": image_bgr} if collect_debug else None

        # Stage 1: quality gate (runs the single landmark pass internally).
        quality = self.quality_gate.check(image_bgr)
        quality_block = {
            "accepted": quality.accepted,
            "pose_ok": not any(r.startswith("pose") for r in quality.reasons),
            "lighting_ok": not any(
                r in ("too_dark", "overexposed", "heavy_glare") for r in quality.reasons),
            "blur_ok": "too_blurry" not in quality.reasons,
            "face_detected": quality.details.get("face_detected", False),
        }
        if not quality.accepted:
            return self._response("retake_requested", quality_block,
                                  message=quality.retake_message or MSG_POOR_QUALITY), debug

        # Stage 2: landmarks (cached from the quality gate's pass).
        landmarks = self.landmarker.detect(image_bgr)
        if not landmarks.success:
            quality_block["face_detected"] = False
            return self._response("abstain", quality_block, message=MSG_POOR_QUALITY), debug

        # Stage 3a: eye-bag branch.
        eye_bags, eye_decision = self._eye_bag_branch(image_bgr, landmarks, debug)

        # Stage 3b: wrinkle branch.
        wrinkles = self._wrinkle_branch(image_bgr, debug)

        # Stage 4: aggregate decision.
        decision, message = self._decide(eye_decision)
        confounders = {
            "significant_asymmetry": eye_decision.get("significant_asymmetry", False),
            "makeup_detected_or_suspected": False,   # not modelled in this build
        }
        return self._response(decision, quality_block, eye_bags=eye_bags,
                              wrinkles=wrinkles, confounders=confounders, message=message), debug

    # ── branches ───────────────────────────────────────────────────────────
    def _eye_bag_branch(self, image_bgr, landmarks, debug=None):
        if self.eye_bag_session is None:
            return {"left": None, "right": None, "available": False}, {"available": False}

        rois = self.cropper.crop(image_bgr, landmarks)
        if not rois.both_valid:
            return ({"left": None, "right": None, "available": False},
                    {"available": False, "rois_failed": True})

        left = self._run_eye(rois.left.crop)
        right = self._run_eye(rois.right.crop)
        if debug is not None:
            debug["eye_bags"] = {
                "left_bbox": rois.left.bbox_xyxy, "right_bbox": rois.right.bbox_xyxy,
                "left_crop": rois.left.crop, "right_crop": rois.right.crop,
                "left": left, "right": right,
            }
        mean_conf = (left["confidence"] + right["confidence"]) / 2.0
        grade_diff = abs(left["severity_grade"] - right["severity_grade"])
        return (
            {"left": left, "right": right, "available": True},
            {
                "available": True,
                "mean_confidence": mean_conf,
                "significant_asymmetry": grade_diff >= ASYMMETRY_GRADE_DIFF,
            },
        )

    def _run_eye(self, crop_bgr: np.ndarray) -> Dict[str, Any]:
        x = preprocess_eye_crop(crop_bgr)
        presence, severity, _dc = self.eye_bag_session.run(None, {self.eye_bag_input_name: x})
        presence_prob = float(_sigmoid(np.asarray(presence).reshape(-1)[0]))
        sev = np.asarray(severity).reshape(-1)
        grade = coral_grade(sev)
        confidence = coral_confidence(sev)
        return {
            "present_probability": round(presence_prob, 3),
            "severity_grade": grade,
            "severity_label": SEVERITY_LABELS.get(grade, str(grade)),
            "confidence": round(confidence, 3),
        }

    def _wrinkle_branch(self, image_bgr, debug=None) -> Dict[str, Any]:
        if self.wrinkle_analyzer is None:
            return empty_wrinkle_result()
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        all_px = self.landmarker.all_points_px(image_bgr)
        result = self.wrinkle_analyzer.analyze(
            rgb, landmarks_px=all_px, keep_mask=debug is not None)
        if debug is not None:
            debug["wrinkles"] = {
                "mask": result.mask,
                "crop_rgb": result.crop_rgb,
                "masked_rgb": result.masked_rgb,
                "texture": result.texture,
                "face_mask": result.face_mask,
                "landmarks_crop": result.landmarks_crop,
                "regions": result.regions,
            }
        return result.to_dict()

    # ── decision ─────────────────────────────────────────────────────────────
    def _decide(self, eye_decision: Dict[str, Any]):
        if eye_decision.get("available"):
            if eye_decision.get("significant_asymmetry"):
                return "abstain", MSG_ASYMMETRY
            if eye_decision.get("mean_confidence", 1.0) < MIN_MEAN_CONFIDENCE:
                return "abstain", MSG_POOR_QUALITY
            return "show_guidance", MSG_FOOTER
        # Eye-bag branch unavailable: still surface wrinkles if we got this far.
        return "show_guidance", MSG_FOOTER

    # ── response assembly ────────────────────────────────────────────────────
    @staticmethod
    def _response(decision, quality, *, eye_bags=None, wrinkles=None,
                  confounders=None, message=""):
        if eye_bags is None:
            eye_bags = {"left": None, "right": None, "available": False}
        if wrinkles is None:
            wrinkles = empty_wrinkle_result()
        return {
            "schema_version": SCHEMA_VERSION,
            "quality": quality,
            "eye_bags": eye_bags,
            "wrinkles": wrinkles,
            "confounders": confounders or {
                "significant_asymmetry": None, "makeup_detected_or_suspected": None},
            "decision": decision,
            "message": message,
            "disclaimer": DISCLAIMER,
        }
