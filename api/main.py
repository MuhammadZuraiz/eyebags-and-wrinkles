#!/usr/bin/env python3
"""
DermaLens inference API (Day 10).

Endpoints:
    GET  /health              — liveness check, returns model status
    POST /analyze/under-eye   — analyse one face image

POST /analyze/under-eye accepts EITHER:
    1. multipart/form-data with a "file" field (an image upload) — easiest
       to test from React Native and from curl, OR
    2. application/json with {"image_base64": "<base64 jpeg/png>"}

Both return the spec's output contract JSON.

Run locally:
    export DERMALENS_CHECKPOINT=experiments/multitask/best.pt
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Test:
    curl -X POST http://localhost:8000/analyze/under-eye \
         -F "file=@selfie.jpg" | python -m json.tool

    # or base64:
    python - << 'PY'
    import base64, requests
    b64 = base64.b64encode(open("selfie.jpg","rb").read()).decode()
    r = requests.post("http://localhost:8000/analyze/under-eye",
                      json={"image_base64": b64})
    print(r.json())
    PY

Interactive docs: http://localhost:8000/docs  (FastAPI auto-generates them)

PRIVACY NOTE (matches the blueprint's Phase A requirements):
    - The image is processed in memory and NEVER written to disk.
    - Only the derived result is returned; nothing is stored server-side.
    - Add TLS termination (e.g. behind nginx/cloud load balancer) before
      any real user traffic — facial images must be encrypted in transit.
"""

import base64
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dermalens.api")

# ── App + model bootstrap ───────────────────────────────────────────────────

app = FastAPI(
    title="DermaLens Under-Eye Analysis API",
    description="Visible-concern analysis of under-eye bags. "
                "Cosmetic skincare guidance, not a medical diagnosis.",
    version="0.1.0-prototype",
)

CHECKPOINT = os.environ.get("DERMALENS_CHECKPOINT", "experiments/multitask/best.pt")
_pipeline = None   # lazy-loaded on startup (torch eye-bag pipeline)

# Unified face-skin analyzer (on-device ONNX: eye bags + wrinkles). Configured
# via env vars; loaded only if at least one ONNX model is provided.
EYE_BAG_ONNX = os.environ.get("DERMALENS_EYE_BAG_ONNX")
WRINKLE_ONNX = os.environ.get("DERMALENS_WRINKLE_ONNX")
LANDMARKER_MODEL = os.environ.get("MEDIAPIPE_FACE_LANDMARKER_MODEL")
_face_analyzer = None


@app.on_event("startup")
def load_pipeline():
    global _pipeline, _face_analyzer
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Legacy torch eye-bag pipeline is optional — it needs torch/torchvision and a
    # .pt checkpoint. Guard it so the on-device ONNX analyzer (below) can still
    # boot in a torch-free / ONNX-only deployment.
    try:
        from src.deployment.inference import DermaLensPipeline
        if not os.path.exists(CHECKPOINT):
            logger.warning(f"Legacy checkpoint not found ({CHECKPOINT}); skipping torch pipeline.")
        else:
            _pipeline = DermaLensPipeline(CHECKPOINT)
            logger.info("Eye-bag (torch) pipeline loaded and ready.")
    except Exception as exc:
        logger.warning(f"Legacy torch pipeline unavailable ({exc}); ONNX analyzer only.")

    # Unified ONNX analyzer (optional) — eye bags + wrinkles.
    if EYE_BAG_ONNX or WRINKLE_ONNX:
        try:
            from src.face_analysis import FaceSkinAnalyzer
            _face_analyzer = FaceSkinAnalyzer.from_paths(
                eye_bag_onnx=EYE_BAG_ONNX,
                wrinkle_onnx=WRINKLE_ONNX,
                landmarker_model=LANDMARKER_MODEL,
            )
            logger.info("Unified face-skin analyzer (ONNX) loaded and ready.")
        except Exception as exc:   # pragma: no cover - depends on provisioned assets
            logger.error(f"Failed to load unified face analyzer: {exc}")


# ── Schemas ─────────────────────────────────────────────────────────────────

class Base64Request(BaseModel):
    image_base64: str


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":             "ok" if (_pipeline or _face_analyzer) else "model_not_loaded",
        "checkpoint":         CHECKPOINT,
        "model_loaded":       _pipeline is not None,
        "face_analyzer_loaded": _face_analyzer is not None,
    }


def _decode_bytes_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image bytes")
    # Guard against absurdly large uploads (decompression bombs)
    if img.shape[0] * img.shape[1] > 40_000_000:   # > 40 MP
        raise HTTPException(status_code=413, detail="Image too large")
    return img


@app.post("/analyze/under-eye")
async def analyze_under_eye(
    file: Optional[UploadFile] = File(default=None),
    body: Optional[Base64Request] = None,
):
    """
    Analyse one face image for visible under-eye bags.

    Provide EITHER a multipart "file" upload OR a JSON body with image_base64.
    Returns the output contract defined in DERMALENS_EYE_BAG_MODEL_SPEC.md §2.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded — check server logs")

    if file is not None:
        data = await file.read()
    elif body is not None and body.image_base64:
        try:
            data = base64.b64decode(body.image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 payload")
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide a multipart 'file' or JSON {'image_base64': ...}",
        )

    if len(data) > 15 * 1024 * 1024:   # 15 MB upload cap
        raise HTTPException(status_code=413, detail="Upload exceeds 15 MB")

    img = _decode_bytes_to_bgr(data)
    result = _pipeline.analyze(img)
    return result


async def _read_image_payload(file: Optional[UploadFile], body: Optional["Base64Request"]) -> bytes:
    if file is not None:
        data = await file.read()
    elif body is not None and body.image_base64:
        try:
            data = base64.b64decode(body.image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 payload")
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide a multipart 'file' or JSON {'image_base64': ...}",
        )
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Upload exceeds 15 MB")
    return data


@app.post("/analyze/face")
async def analyze_face(
    file: Optional[UploadFile] = File(default=None),
    body: Optional[Base64Request] = None,
):
    """
    Unified analysis of one selfie for eye bags + wrinkles (on-device ONNX path).

    Returns the unified contract from src.face_analysis.pipeline (schema_version,
    quality, eye_bags, wrinkles, decision, message, disclaimer). Dark circles are
    intentionally out of scope in this build.
    """
    if _face_analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Face analyzer not loaded — set DERMALENS_EYE_BAG_ONNX / "
                   "DERMALENS_WRINKLE_ONNX (and MEDIAPIPE_FACE_LANDMARKER_MODEL).",
        )
    data = await _read_image_payload(file, body)
    img = _decode_bytes_to_bgr(data)
    return _face_analyzer.analyze(img)


@app.post("/analyze/face/debug")
async def analyze_face_debug(
    file: Optional[UploadFile] = File(default=None),
    body: Optional[Base64Request] = None,
):
    """
    Same as /analyze/face, but also returns rendered overlay images (PNG
    data-URIs) so you can visually verify what each model focused on:
    eye-bag ROI boxes + crops, wrinkle segmentation mask, region polygons, and
    the texture-map input. Returns {"result": <unified JSON>, "overlays": {...}}.
    """
    if _face_analyzer is None:
        raise HTTPException(status_code=503, detail="Face analyzer not loaded.")
    from src.face_analysis.visualize import render_overlays

    data = await _read_image_payload(file, body)
    img = _decode_bytes_to_bgr(data)
    result, debug = _face_analyzer.analyze_debug(img)
    overlays = render_overlays(debug)
    return {"result": result, "overlays": overlays}


# ── Frontend (verification UI) ───────────────────────────────────────────────

_WEB_DIR = Path(__file__).parent.parent / "web"


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = _WEB_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>UI not found</h1><p>web/index.html is missing.</p>", status_code=404)
    return FileResponse(str(index_path))
