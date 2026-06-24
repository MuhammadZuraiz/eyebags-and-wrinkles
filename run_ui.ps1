# Launch the eye-bag + wrinkle verification UI (Windows / PowerShell).
# Prereqs (see SETUP.md): .venv created + deps installed, and the model files
# dropped into .\models\. Then:  .\run_ui.ps1   and open http://localhost:8000
$root = $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "No .venv found. See SETUP.md - create the venv and install deps first." -ForegroundColor Red
    exit 1
}

$models = Join-Path $root "models"
$eye  = Join-Path $models "model.onnx"
$wr   = Join-Path $models "wrinkle_unet.onnx"
$task = Join-Path $models "face_landmarker.task"

if (Test-Path $eye)  { $env:DERMALENS_EYE_BAG_ONNX = $eye }
else { Write-Host "note: models\model.onnx missing - eye-bag analysis disabled" -ForegroundColor Yellow }
if (Test-Path $wr)   { $env:DERMALENS_WRINKLE_ONNX = $wr }
else { Write-Host "note: models\wrinkle_unet.onnx missing - wrinkle analysis disabled" -ForegroundColor Yellow }
if (Test-Path $task) { $env:MEDIAPIPE_FACE_LANDMARKER_MODEL = $task }
else { Write-Host "ERROR: models\face_landmarker.task missing - required." -ForegroundColor Red; exit 1 }

$env:DERMALENS_CHECKPOINT = "none"   # skip the optional legacy torch pipeline
Write-Host "Starting UI at http://127.0.0.1:8000  (Ctrl+C to stop)" -ForegroundColor Green
& $py -m uvicorn api.main:app --host 127.0.0.1 --port 8000
