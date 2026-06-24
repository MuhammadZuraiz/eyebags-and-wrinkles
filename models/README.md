# models/

Drop the three model files here. They are **not** in git (weights are gitignored
for size/license/privacy). The launch scripts (`run_ui.ps1` / `run_ui.sh`) read
them from this folder.

| File | What | Where to get it |
|------|------|-----------------|
| `face_landmarker.task` | MediaPipe Tasks face landmarker | `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task` |
| `model.onnx` **+** `model.onnx.data` | trained eye-bag model | your team's model storage (shared drive / bucket / release). **Copy both files** — `.onnx` needs its `.onnx.data` sidecar. |
| `wrinkle_unet.onnx` **+** `wrinkle_unet.onnx.data` | wrinkle U-Net | export it yourself (see SETUP.md): `python -m src.wrinkles.export_onnx --checkpoint stage2_unet.pth --output models/wrinkle_unet.onnx` |

Notes
- Either model may be omitted — the corresponding block is reported as
  "unavailable" and the other still runs. `face_landmarker.task` is always required.
- `stage2_unet.pth` (the wrinkle checkpoint) comes from the labhai FFHQ-Wrinkle
  Google Drive; keep it anywhere, it's only needed for the one-time ONNX export.
