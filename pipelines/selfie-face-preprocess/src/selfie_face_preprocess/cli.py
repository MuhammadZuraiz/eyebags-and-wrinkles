from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PreprocessConfig
from .mediapipe_adapter import MediaPipeConfigurationError
from .pipeline import preprocess_selfie


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skin-preprocess",
        description="Preprocess a selfie into an aligned face-only skin ML input.",
    )
    parser.add_argument("input_image", help="Path to a selfie image.")
    parser.add_argument("--out-dir", default="processed", help="Directory for output PNG/JSON files.")
    parser.add_argument("--size", type=int, default=1024, help="Square output size in pixels.")
    parser.add_argument("--face-landmarker-model", help="Path to MediaPipe face_landmarker.task.")
    parser.add_argument("--face-detector-model", help="Optional path to MediaPipe face_detector.task.")
    parser.add_argument(
        "--allow-rejected",
        action="store_true",
        help="Return exit code 0 even when the image is rejected.",
    )
    parser.add_argument(
        "--no-quality-reject",
        action="store_true",
        help="Record quality problems as warnings instead of reject reasons.",
    )
    parser.add_argument(
        "--model-input-mode",
        choices=("aligned", "masked_fill"),
        default="aligned",
        help=(
            "Use 'aligned' for a natural face crop plus mask sidecar, or "
            "'masked_fill' to fill non-face pixels with the median face color."
        ),
    )
    parser.add_argument("--min-face-fraction", type=float, default=0.18)
    parser.add_argument("--blur-threshold", type=float, default=45.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = PreprocessConfig(
        output_size=args.size,
        min_face_fraction=args.min_face_fraction,
        blur_threshold=args.blur_threshold,
        reject_on_quality=not args.no_quality_reject,
        model_input_mode=args.model_input_mode,
    )
    if args.face_landmarker_model:
        config.face_landmarker_model_path = args.face_landmarker_model
    if args.face_detector_model:
        config.face_detector_model_path = args.face_detector_model

    try:
        result = preprocess_selfie(args.input_image, config)
    except MediaPipeConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    written = result.save_outputs(Path(args.out_dir))
    summary = {
        "accepted": result.accepted,
        "model_input_mode": result.metadata.get("config", {}).get("model_input_mode"),
        "reject_reasons": result.reject_reasons,
        "warnings": result.warnings,
        "outputs": {key: str(path) for key, path in written.items()},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if result.accepted or args.allow_rejected:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
