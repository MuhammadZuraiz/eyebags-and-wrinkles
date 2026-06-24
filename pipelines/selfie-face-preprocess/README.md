# Selfie Face Preprocessing Pipeline

This package converts a phone selfie into a normalized, face-only ML input for skin issue detection. It is conservative by design: it aligns and masks the face, improves capture quality lightly, and returns structured reject reasons instead of trying to repair unusable images.

## Install

```bash
python -m pip install -e ".[full]"
```

MediaPipe currently publishes the most reliable wheels for Python 3.10-3.12. If you are on a newer Python, install the base package for tests and utility code, then run the full MediaPipe pipeline in a Python 3.10-3.12 environment.

The full pipeline requires a MediaPipe Face Landmarker task model:

```bash
skin-preprocess selfie.jpg --out-dir processed --face-landmarker-model /absolute/path/to/face_landmarker.task
```

For most skin-issue models, use the default `aligned` model input. It preserves the real aligned face crop and writes `face_mask.png` separately. Use `--model-input-mode masked_fill` only when the downstream model was explicitly trained on flat-background face cutouts.

You can also set:

```bash
MEDIAPIPE_FACE_LANDMARKER_MODEL=/absolute/path/to/face_landmarker.task
MEDIAPIPE_FACE_DETECTOR_MODEL=/absolute/path/to/face_detector.task
```

The detector model is optional. If supplied, it is used to catch multiple faces before landmarking.

## Sharing With a Team

Commit the package source, tests, README, `pyproject.toml`, and `.gitignore` to your application or ML repo. Do not commit local virtual environments, downloaded `.task` model files, generated outputs, or real user selfies.

Recommended repo layout:

```text
repo/
  pyproject.toml
  README.md
  .gitignore
  src/selfie_face_preprocess/
  tests/
```

Each teammate should create their own virtual environment, install the package with `python -m pip install -e ".[full]"`, and download or otherwise provision the MediaPipe `face_landmarker.task` model locally. For production, store model files in your artifact registry, cloud bucket, release asset storage, or model registry rather than Git.

## Outputs

For each processed image, the CLI writes:

- `model_input.png`: aligned square RGB face image. By default this is the natural aligned crop; pass `--model-input-mode masked_fill` to fill non-face pixels with the median face color.
- `face_aligned.png`: unmasked aligned crop for audit/debug review.
- `face_mask.png`: binary face oval mask.
- `metadata.json`: quality scores, reject reasons, warnings, transform, landmarks, and preprocessing parameters.

The command exits with status `0` for accepted inputs and `2` for rejected inputs. Use `--allow-rejected` when batch jobs should continue with a zero exit code while still recording reject reasons in metadata.

## Python API

```python
from selfie_face_preprocess import PreprocessConfig, preprocess_selfie

config = PreprocessConfig(
    output_size=1024,
    face_landmarker_model_path="/absolute/path/to/face_landmarker.task",
)

result = preprocess_selfie("selfie.jpg", config)
result.save_outputs("processed")

if not result.accepted:
    print(result.reject_reasons)
```

## Design Notes

- No beauty filters, acne removal, aggressive smoothing, or hallucinated repair.
- Rejects bad captures with reasons such as `no_face`, `multiple_faces`, `face_too_small`, `extreme_pose`, `too_blurry`, `too_dark`, `too_bright`, `heavy_glare`, and `landmarks_failed`.
- Writes a face-oval mask sidecar, not a skin-only segmentation mask.
- Applies only mild exposure normalization, capped gray-world white balance, low-strength CLAHE, and optional light denoise when quality scores indicate it is needed.
