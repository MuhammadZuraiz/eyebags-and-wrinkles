#!/usr/bin/env python3
"""
End-to-end inference pipeline for DermaLens.

Implements the FULL flow from the spec (Section 9):

    Full face image
        ↓ Quality gate
        ↓ MediaPipe landmarks
        ↓ Left + right under-eye crops
        ↓ Multi-task model (run once per eye)
        ↓ Abstention + asymmetry rules (spec Section 6)
        ↓ Output JSON exactly matching the spec's contract (Section 2)

Decision logic (spec Section 6 — abstention rules):
    decision = "retake_requested"  if the quality gate rejects the image
    decision = "abstain"           if landmarks fail, mean confidence < 0.40,
                                   or left-right grade difference ≥ 2
    decision = "show_guidance"     otherwise

Usage:
    from src.deployment.inference import DermaLensPipeline

    pipeline = DermaLensPipeline("experiments/multitask/best.pt")
    result   = pipeline.analyze_image_file("selfie.jpg")
    print(json.dumps(result, indent=2))
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch

from src.preprocessing.quality_gate   import QualityGate
from src.preprocessing.face_landmarks import FaceLandmarker
from src.preprocessing.roi_cropper    import RoiCropper
from src.data.augmentations           import get_val_transforms
from src.models.multitask             import load_model_from_checkpoint
from src.models.ordinal_head          import coral_logits_to_grade, coral_grade_confidence

logger = logging.getLogger(__name__)

SEVERITY_LABELS = {
    0: "not_present",
    1: "mild",
    2: "moderate",
    3: "pronounced",
    4: "very_pronounced",
}

# Spec Section 6 thresholds
MIN_MEAN_CONFIDENCE  = 0.40
ASYMMETRY_GRADE_DIFF = 2

# Spec Section 7 — exact user-facing strings
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


class DermaLensPipeline:
    """
    Loads everything once at startup; call .analyze() per image.

    Args:
        checkpoint_path: Path to the trained model checkpoint (best.pt).
        device:          "cuda", "cpu", or None for auto-detect.
    """

    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ── Load model (architecture from checkpoint model_config) ────────
        self.model = load_model_from_checkpoint(checkpoint_path, self.device)

        self.has_severity = self.model.severity_head is not None
        self.has_dc       = self.model.dark_circles_head is not None

        # ── Preprocessing components (shared landmarker → one model load) ──
        self.landmarker   = FaceLandmarker()
        self.quality_gate = QualityGate(landmarker=self.landmarker)
        self.cropper      = RoiCropper()
        self.transform    = get_val_transforms()

        logger.info(
            f"DermaLensPipeline ready: {checkpoint_path} on {self.device}  "
            f"(severity={self.has_severity}, dark_circles={self.has_dc})"
        )

    # ─────────────────────────────────────────────────────────────────────────

    def analyze_image_file(self, image_path: str) -> Dict:
        """Convenience wrapper: load from disk then analyse."""
        img = cv2.imread(str(image_path))
        if img is None:
            return self._error_response(f"Could not read image: {image_path}")
        return self.analyze(img)

    @torch.no_grad()
    def analyze(self, image_bgr: np.ndarray) -> Dict:
        """
        Full pipeline on one BGR face image.

        Returns the spec's output contract JSON (Section 2) as a Python dict.
        """
        # ── Stage 1: quality gate ─────────────────────────────────────────
        quality = self.quality_gate.check(image_bgr)
        quality_block = {
            "accepted":      quality.accepted,
            "pose_ok":       not any(r.startswith("pose") for r in quality.reasons),
            "lighting_ok":   not any(r in ("too_dark", "overexposed", "heavy_glare") for r in quality.reasons),
            "blur_ok":       "too_blurry" not in quality.reasons,
            "face_detected": quality.details.get("face_detected", False),
        }

        if not quality.accepted:
            return self._response(
                decision="retake_requested",
                quality=quality_block,
                message=quality.retake_message or MSG_POOR_QUALITY,
            )

        # ── Stage 2: landmarks + crops ────────────────────────────────────
        landmarks = self.landmarker.detect(image_bgr)
        if not landmarks.success:
            quality_block["face_detected"] = False
            return self._response(
                decision="abstain", quality=quality_block, message=MSG_POOR_QUALITY
            )

        rois = self.cropper.crop(image_bgr, landmarks)
        if not rois.both_valid:
            return self._response(
                decision="abstain", quality=quality_block, message=MSG_POOR_QUALITY
            )

        # ── Stage 3: model inference (batch both eyes in one forward pass) ──
        left_result  = self._run_model(rois.left.crop)
        right_result = self._run_model(rois.right.crop)

        # ── Stage 4: abstention + asymmetry rules (spec Section 6) ────────
        mean_conf = (left_result["confidence"] + right_result["confidence"]) / 2.0
        grade_diff = abs(left_result["severity_grade"] - right_result["severity_grade"])

        significant_asymmetry = grade_diff >= ASYMMETRY_GRADE_DIFF

        dark_circles_visible = bool(
            left_result.get("dc_prob", 0) > 0.5 or right_result.get("dc_prob", 0) > 0.5
        )

        confounders = {
            "dark_circles_visible":         dark_circles_visible,
            "makeup_detected_or_suspected": False,   # not modelled in prototype
            "significant_asymmetry":        significant_asymmetry,
        }

        if significant_asymmetry:
            decision, message = "abstain", MSG_ASYMMETRY
        elif mean_conf < MIN_MEAN_CONFIDENCE:
            decision, message = "abstain", MSG_POOR_QUALITY
        else:
            decision, message = "show_guidance", MSG_FOOTER

        return self._response(
            decision=decision,
            quality=quality_block,
            confounders=confounders,
            left=left_result,
            right=right_result,
            message=message,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _run_model(self, crop_bgr: np.ndarray) -> Dict:
        """Run the model on a single 256×160 BGR crop."""
        from PIL import Image
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor   = self.transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(self.device)

        out = self.model(tensor)

        presence_prob = float(torch.sigmoid(out["presence_logit"]).item())

        if self.has_severity:
            sev_logits = out["severity_logits"].cpu()
            grade      = int(coral_logits_to_grade(sev_logits).item())
            confidence = float(coral_grade_confidence(sev_logits).item())
        else:
            grade      = 1 if presence_prob > 0.5 else 0
            confidence = abs(presence_prob - 0.5) * 2.0

        result = {
            "present_probability": round(presence_prob, 3),
            "severity_grade":      grade,
            "severity_label":      SEVERITY_LABELS[grade],
            "confidence":          round(confidence, 3),
        }
        if self.has_dc:
            result["dc_prob"] = float(torch.sigmoid(out["dark_circles_logit"]).item())
        return result

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _response(
        decision:    str,
        quality:     Dict,
        confounders: Optional[Dict] = None,
        left:        Optional[Dict] = None,
        right:       Optional[Dict] = None,
        message:     str = "",
    ) -> Dict:
        """Assemble the output contract (spec Section 2)."""
        def _eye(r: Optional[Dict]) -> Optional[Dict]:
            if r is None:
                return None
            return {k: v for k, v in r.items() if k != "dc_prob"}

        return {
            "eye_bags": {
                "left":  _eye(left),
                "right": _eye(right),
            },
            "confounders": confounders or {
                "dark_circles_visible":         None,
                "makeup_detected_or_suspected": None,
                "significant_asymmetry":        None,
            },
            "quality":  quality,
            "decision": decision,
            "message":  message,
        }

    @staticmethod
    def _error_response(msg: str) -> Dict:
        return {
            "eye_bags":    {"left": None, "right": None},
            "confounders": {},
            "quality":     {"accepted": False, "face_detected": False},
            "decision":    "retake_requested",
            "message":     msg,
        }
