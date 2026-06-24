#!/usr/bin/env bash
# Launch the eye-bag + wrinkle verification UI (macOS / Linux).
# Prereqs (see SETUP.md): .venv created + deps installed, and the model files
# dropped into ./models/. Then:  ./run_ui.sh   and open http://localhost:8000
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py="$root/.venv/bin/python"
[ -x "$py" ] || { echo "No .venv found. See SETUP.md - create the venv and install deps first."; exit 1; }

models="$root/models"
if [ -f "$models/model.onnx" ]; then export DERMALENS_EYE_BAG_ONNX="$models/model.onnx"
else echo "note: models/model.onnx missing - eye-bag analysis disabled"; fi
if [ -f "$models/wrinkle_unet.onnx" ]; then export DERMALENS_WRINKLE_ONNX="$models/wrinkle_unet.onnx"
else echo "note: models/wrinkle_unet.onnx missing - wrinkle analysis disabled"; fi
if [ -f "$models/face_landmarker.task" ]; then export MEDIAPIPE_FACE_LANDMARKER_MODEL="$models/face_landmarker.task"
else echo "ERROR: models/face_landmarker.task missing - required."; exit 1; fi

export DERMALENS_CHECKPOINT="none"   # skip the optional legacy torch pipeline
echo "Starting UI at http://127.0.0.1:8000  (Ctrl+C to stop)"
exec "$py" -m uvicorn api.main:app --host 127.0.0.1 --port 8000
