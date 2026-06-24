# DermaLens — Eye-bag &amp; Wrinkle Analysis

Single-selfie detection of under-eye bags **and** wrinkles, with a verification
web UI that overlays exactly what each model focused on. Cosmetic skincare
guidance only — **not a medical device**.

> 👉 **Running it for the first time? See [SETUP.md](SETUP.md)** (Python 3.10–3.12,
> venv, the three model files, then `run_ui.ps1` / `run_ui.sh`).

> **Not in version control** (see `.gitignore`): face image data
> (`data/raw`, `data/crops`, `data/external`), model weights (`*.pt`, ONNX) and
> Colab run bundles are excluded for size/license/privacy. They live in external
> storage (cloud bucket / model registry / GitHub Release). What *is* committed:
> all code, configs, tests, docs, the synthetic `data/smoke` set, and the label
> CSVs (`data/annotations`, `data/splits`). Final metrics are in
> [docs/MODEL_CARD.md](docs/MODEL_CARD.md); reproduce data via the scripts in
> `scripts/` (`fetch_london_set.py`, `build_ffhq_subset.py`).

## What's here

```
dermalens-eye-bags/
├── configs/            # YAML configs for the 3 training stages (Days 5/7/8)
├── api/main.py         # FastAPI inference endpoint
├── docs/
│   ├── WALKTHROUGH.md          ← START HERE: day-by-day exact steps
│   ├── annotation_guide.md     # Label Studio setup + severity rubric
│   ├── MODEL_CARD_TEMPLATE.md
│   └── DERMALENS_EYE_BAG_MODEL_SPEC.md
├── notebooks/
│   └── colab_train.ipynb       # all training runs (free T4) — see docs/COLAB.md
├── scripts/
│   ├── fetch_london_set.py     # seed data: London Set (CC BY 4.0) from figshare
│   ├── build_ffhq_subset.py    # scale data: license-filtered, age-skewed FFHQ
│   ├── batch_crop.py           # faces → under-eye crops (quality-gated)
│   ├── generate_ls_tasks.py    # crops + manifests → Label Studio tasks
│   ├── annotation_qa.py        # two-pass self-consistency gate (kappa)
│   ├── prepare_training_csv.py # Label Studio export → training CSV
│   ├── make_colab_bundle.py    # zip code+splits+crops for Colab
│   ├── train.py                # train any stage from config (--resume support)
│   ├── evaluate.py             # milestone-gate metrics + subgroup reports
│   └── error_analysis.py       # worst-mistake image grids
├── src/
│   ├── preprocessing/  # MediaPipe landmarks, ROI cropper, quality gate
│   ├── data/           # dataset, subject-level splits, augmentations, sampler
│   ├── models/         # CORAL ordinal head, multi-task model
│   ├── training/       # losses, trainer (AMP, cosine LR, early stop, resume)
│   ├── evaluation/     # metrics incl. QWK + subgroup fairness report
│   └── deployment/     # end-to-end inference pipeline, ONNX export
└── tests/              # 33 tests: ordinal math, leakage, checkpoints, contract
```

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # test suite passed = environment is good

# Verify the whole pipeline on synthetic data (no real data needed):
python -m src.data.splits --annotations data/smoke/annotations.csv --output data/smoke/splits
python scripts/train.py --config configs/smoke.yaml
python scripts/error_analysis.py --checkpoint experiments/smoke/best.pt --csv data/smoke/splits/val.csv
python src/deployment/export_onnx.py --checkpoint experiments/smoke/best.pt --output experiments/smoke/model.onnx
```

Then follow **docs/WALKTHROUGH.md** with your real data.

Dataset note: public/free source material is for prototype training only.
Training CSV rows must include non-empty `subject_id`, `source_dataset`, and
`license_status` so subject-level splits and source/license audits are possible.

## Serve the API

```bash
export DERMALENS_CHECKPOINT=experiments/multitask/best.pt
uvicorn api.main:app --port 8000
curl -X POST localhost:8000/analyze/under-eye -F "file=@selfie.jpg"
```

## Unified face-skin analysis (eye bags + wrinkles)

`src/face_analysis/` adds a single-selfie pipeline that runs **eye bags** (this
project's model) and **wrinkles** (a vendored U-Net) behind one MediaPipe Tasks
landmark pass, on a torch-free **onnxruntime** runtime for on-device use. Dark
circles are intentionally out of scope in this build.

```
selfie ─▶ quality gate ─▶ MediaPipe Tasks landmarks (one pass)
                              ├─ eye bags : ROI crops ─▶ eye_bag.onnx (per eye)
                              └─ wrinkles : masked 1024 face + texture map ─▶ wrinkle_unet.onnx
                          ─▶ unified JSON {quality, eye_bags, wrinkles, decision, message}
```

**Provision the assets** (none are in git):

| Asset | What | How |
|---|---|---|
| `eye_bag.onnx` | eye-bag model | `python src/deployment/export_onnx.py --checkpoint experiments/ordinal_severity/best.pt --output experiments/ordinal_severity/model.onnx` |
| `wrinkle_unet.onnx` | wrinkle U-Net | `python -m src.wrinkles.export_onnx --checkpoint stage2_unet.pth --output experiments/wrinkles/wrinkle_unet.onnx` |
| `face_landmarker.task` | MediaPipe Tasks model | download from MediaPipe; or set `MEDIAPIPE_FACE_LANDMARKER_MODEL` |

`stage2_unet.pth` is the labhai FFHQ-Wrinkle checkpoint (their Google Drive). The
vendored `src/wrinkles/unet/` is **GPLv3** (see its headers) — that license
applies to those files and anything linked against them.

**Run the CLI:**

```bash
python scripts/face_analyze.py selfie.jpg \
    --eye-bag-onnx experiments/ordinal_severity/model.onnx \
    --wrinkle-onnx experiments/wrinkles/wrinkle_unet.onnx \
    --landmarker-model face_landmarker.task
# exit 0 = analysed (show_guidance/abstain), 2 = retake_requested
```

**Or via the API** (`POST /analyze/face`):

```bash
export DERMALENS_EYE_BAG_ONNX=experiments/ordinal_severity/model.onnx
export DERMALENS_WRINKLE_ONNX=experiments/wrinkles/wrinkle_unet.onnx
export MEDIAPIPE_FACE_LANDMARKER_MODEL=face_landmarker.task
uvicorn api.main:app --port 8000
curl -X POST localhost:8000/analyze/face -F "file=@selfie.jpg"
```

Needs `pip install -e pipelines/selfie-face-preprocess` (the shared front-end)
plus `onnxruntime` and `mediapipe`. Either ONNX model may be omitted; its block
is then reported as unavailable.

## Fairness audit snippet

```python
import pandas as pd, torch
from src.evaluation.metrics import subgroup_report
# after collecting logits/targets on a test loader (see Trainer.evaluate):
report = subgroup_report(df_meta, presence_logits, presence_targets,
                         severity_logits, severity_targets, group_col="mst_shade")
print(report["_gaps"])   # sensitivity_gap target ≤ 0.05
```

## Key design decisions
- **CORAL ordinal head** — grade errors penalised by distance, rank-consistent by construction
- **Subject-level splits** — same person never in train AND test (the #1 silent killer)
- **Dark-circles auxiliary head** (weight 0.25) — forces encoder to separate puffiness from pigmentation
- **Abstention layer** — low confidence or left/right asymmetry ≥ 2 grades → no guidance shown
- **Orientation-normalised crops** — outer corner always left; one model handles both eyes
