# Team Setup

How to run the eye-bag + wrinkle analyzer + verification UI from a fresh clone.

## 0. Prerequisites
- **Python 3.10, 3.11, or 3.12** — *not* 3.13+. MediaPipe (pinned `0.10.14`)
  only ships wheels for 3.10–3.12; newer Pythons will fail to install it.
- Git. (Windows users: the launcher is PowerShell; macOS/Linux use the `.sh`.)

Check your version: `py -3.12 --version` (Windows) or `python3.12 --version`.

## 1. Clone
```bash
git clone https://github.com/MuhammadZuraiz/eyebags-and-wrinkles.git
cd eyebags-and-wrinkles
```

## 2. Create the venv + install deps

**Windows (PowerShell):**
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e pipelines/selfie-face-preprocess
```

**macOS / Linux:**
```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install -e pipelines/selfie-face-preprocess
```

`requirements.txt` pulls torch (CPU is fine — only used for the one-time ONNX
export), onnxruntime, mediapipe, opencv, fastapi/uvicorn. `selfie-face-preprocess`
is the shared MediaPipe front-end, bundled in this repo.

## 3. Get the three model files into `models/`
See [models/README.md](models/README.md). In short:

1. **Landmarker** (required):
   ```bash
   curl -L -o models/face_landmarker.task \
     https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
   ```
2. **Eye-bag model**: copy `model.onnx` **and** `model.onnx.data` from your
   team's model storage into `models/`.
3. **Wrinkle model**: get `stage2_unet.pth` (labhai FFHQ-Wrinkle Drive), then
   export once:
   ```bash
   # Windows:  .\.venv\Scripts\python.exe -m src.wrinkles.export_onnx ...
   ./.venv/bin/python -m src.wrinkles.export_onnx \
       --checkpoint /path/to/stage2_unet.pth \
       --output models/wrinkle_unet.onnx
   ```
   It prints `VALIDATED (< 1e-4)` on success and writes
   `wrinkle_unet.onnx` + `wrinkle_unet.onnx.data`.

## 4. Run

**UI (recommended for verifying overlays):**
```bash
# Windows:
.\run_ui.ps1
# macOS/Linux:
chmod +x run_ui.sh && ./run_ui.sh
```
Open <http://localhost:8000>, drop a **front-facing, evenly lit** selfie, and hit
Analyze. You'll see the decision, scores, and overlays showing exactly what each
model focused on (ROI boxes, wrinkle mask, region polygons, texture input).

**CLI (one image → JSON):**
```bash
./.venv/bin/python scripts/face_analyze.py selfie.jpg \
  --eye-bag-onnx     models/model.onnx \
  --wrinkle-onnx     models/wrinkle_unet.onnx \
  --landmarker-model models/face_landmarker.task
```

**Tests (no models needed — synthetic):**
```bash
./.venv/bin/python -m pytest tests/ -q
```

## Troubleshooting
- **`mediapipe` won't install** → you're on Python 3.13+. Recreate `.venv` with 3.10–3.12.
- **UI says "face analyzer NOT loaded"** → a model file is missing from `models/`
  (the landmarker is required). Check the server console for which one.
- **Every photo says RETAKE / "face the camera directly"** → the quality gate
  rejects non-frontal / tilted / poorly lit faces by design. Use a straight-on,
  well-lit selfie.
- **`ModuleNotFoundError: selfie_face_preprocess`** → run the
  `pip install -e pipelines/selfie-face-preprocess` step.

> Cosmetic skincare guidance only — **not a medical device, not a diagnosis.**
