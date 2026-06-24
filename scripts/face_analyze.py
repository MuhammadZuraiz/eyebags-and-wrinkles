#!/usr/bin/env python3
"""
CLI: analyse one selfie for eye bags + wrinkles, print the unified JSON.

Runs the on-device (onnxruntime) pipeline. Provide the exported ONNX models and
the MediaPipe Tasks landmarker model:

    python scripts/face_analyze.py selfie.jpg \
        --eye-bag-onnx  experiments/ordinal_severity/model.onnx \
        --wrinkle-onnx  experiments/wrinkles/wrinkle_unet.onnx \
        --landmarker-model face_landmarker.task

Exit codes:
    0  analysed (decision = show_guidance or abstain)
    2  retake_requested (quality gate rejected the image)
    1  configuration/IO error

Either model may be omitted; the corresponding block is reported as unavailable.
The landmarker model can also be supplied via MEDIAPIPE_FACE_LANDMARKER_MODEL.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="face-analyze",
        description="Unified eye-bag + wrinkle analysis for a single selfie.",
    )
    parser.add_argument("input_image", help="Path to a selfie image.")
    parser.add_argument("--eye-bag-onnx", help="Path to the eye-bag model.onnx.")
    parser.add_argument("--wrinkle-onnx", help="Path to wrinkle_unet.onnx.")
    parser.add_argument("--landmarker-model",
                        help="Path to MediaPipe face_landmarker.task "
                             "(or set MEDIAPIPE_FACE_LANDMARKER_MODEL).")
    parser.add_argument("--out", help="Write the JSON result to this file too.")
    parser.add_argument("--allow-rejected", action="store_true",
                        help="Return exit code 0 even when the image is rejected.")
    args = parser.parse_args(argv)

    import cv2
    from src.face_analysis import FaceSkinAnalyzer

    image = cv2.imread(args.input_image)
    if image is None:
        print(f"Configuration error: could not read image: {args.input_image}",
              file=sys.stderr)
        return 1

    if not args.eye_bag_onnx and not args.wrinkle_onnx:
        print("Configuration error: provide --eye-bag-onnx and/or --wrinkle-onnx.",
              file=sys.stderr)
        return 1

    analyzer = FaceSkinAnalyzer.from_paths(
        eye_bag_onnx=args.eye_bag_onnx,
        wrinkle_onnx=args.wrinkle_onnx,
        landmarker_model=args.landmarker_model,
    )
    result = analyzer.analyze(image)

    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")

    if result["decision"] != "retake_requested" or args.allow_rejected:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
