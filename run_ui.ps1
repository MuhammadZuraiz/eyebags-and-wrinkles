# Launch the DermaLens eye-bag + wrinkle verification UI.
# Usage:  .\run_ui.ps1     (from the eye-bags folder)
# Then open http://localhost:8000 in your browser.

$env:DERMALENS_EYE_BAG_ONNX  = "C:\Users\zurai\Desktop\Derma_Lens\Eye bags\experiments\ordinal_severity\model.onnx"
$env:DERMALENS_WRINKLE_ONNX  = "$PSScriptRoot\experiments\wrinkles\wrinkle_unet.onnx"
$env:MEDIAPIPE_FACE_LANDMARKER_MODEL = "C:\Users\zurai\Desktop\Derma_Lens\Wrinkles\face_landmarker.task"
# Skip the optional legacy torch pipeline (we only need the ONNX analyzer).
$env:DERMALENS_CHECKPOINT = "none"

& "$PSScriptRoot\.venv\Scripts\python.exe" -m uvicorn api.main:app --host 127.0.0.1 --port 8000
