# DermaLens Walkthrough — Exact Steps

This is the hand-holding version. Every command, in order, with what success
looks like at each step. The code is already written and tested — your job is
mostly annotation + running scripts + looking at results.

**Where to run things:**
- Data prep + annotation + error analysis: your laptop (no GPU needed).
- Training: Google Colab free T4 GPU (see `docs/COLAB.md`).

**Milestones (agreed quality bar):**
- **Milestone 1:** presence AUROC ≥ 0.90 on held-out subjects
- **Milestone 2:** severity QWK ≥ 0.70 AND within-one ≥ 0.90 on held-out subjects

**One-time setup (laptop, already done if `.venv` exists):**
```powershell
py -3.11 -m venv .venv          # MUST be 3.11/3.12 — mediapipe has no 3.14 wheels
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest tests -q       # should print: 33 passed
```

---

## Step 1 — Get the seed data (DONE — rerun only if data/ is lost)

```powershell
python scripts\fetch_london_set.py
```

Downloads the Face Research Lab London Set (CC BY 4.0, 102 consented adults,
neutral+smiling frontals) and writes `data/raw/london_faces/manifest.csv`.

**Status: complete.** 204 images, 102 subjects, manifest written.

---

## Step 2 — Crop the under-eye regions (DONE for London)

```powershell
python scripts\batch_crop.py --input data\raw\london_faces\images --output data\crops\london --save-overlays
```

**Status: complete.** 197/204 image pairs accepted (3.4% quality-gate
rejections), 394 crops in `data/crops/london/`. Overlays spot-checked — ROI
boxes correctly span lower lid to upper cheek.

If you add a new source, ALWAYS look at 30 overlay images before annotating.

---

## Step 3 — Scale with FFHQ (license-filtered, age-skewed)

The London set skews young (only 16 images age 40+). Eye bags correlate with
age — without older faces, grades 3–4 will be starved and Milestone 2 fails.

```powershell
python scripts\build_ffhq_subset.py ^
    --metadata data\external\ffhq-dataset-v2.json ^
    --features-dir data\external\ffhq-features.zip ^
    --target 700
```

~5,900 of FFHQ's 70k images carry per-image Public Domain / CC0 / US-Gov
licenses; the script selects ~700 skewed old (60% age ≥ 45). Without
`--images-dir` it downloads each selected image from the per-image Drive URLs
(resumable — re-run if quota errors appear). If you have a Kaggle FFHQ mirror,
pass `--images-dir <path>` instead (faster, no quota).

Then crop:
```powershell
python scripts\batch_crop.py --input data\raw\ffhq_subset\images --output data\crops\ffhq --save-overlays
```

Expect a higher rejection rate (20–40%) — these are in-the-wild photos.

**Done when:** ≥ 1,200 total crops across both sources. If under 1,000,
increase `--target` before annotating.

---

## Step 4 — Calibrate yourself, then annotate (the long pole, ~3–5 days)

Label Studio runs in its OWN venv (it pins conflicting dependencies):

```powershell
py -3.11 -m venv .venv-ls
.\.venv-ls\Scripts\pip install label-studio
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED='true'
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT='C:\Users\zurai\Desktop\Derma_Lens\Eye bags\data'
.\.venv-ls\Scripts\label-studio start
```

Create the project per `docs/annotation_guide.md` (paste the XML from section 4).

**4a. Calibration batch first (do not skip):**
1. Import `data/tasks/calibration_tasks.json` (80 crops, already generated).
2. Annotate all 80. Export as JSON → `exports/calib_a.json`.
3. Reset the annotations (or duplicate the project), annotate the same 80
   again in the reshuffled order. Export → `exports/calib_b.json`.
4. Check your self-consistency:
   ```powershell
   python scripts\annotation_qa.py --pass-a exports\calib_a.json --pass-b exports\calib_b.json
   ```
   **Gate: exact kappa ≥ 0.6 OR within-one ≥ 0.8.** If it fails, the script
   shows which grade boundary is noisy — add tiebreak rules to
   `docs/grade_atlas.md`, then redo with 50 fresh crops.
5. Fill `docs/grade_atlas.md` with 3–4 both-passes-agreed exemplars per grade.
   Keep it open during all annotation.

**4b. Full annotation — accelerated (active-learning pre-annotation).**

You do NOT label all ~1,350 crops cold. Instead: trim to a stratified ~800,
hand-label a ~300 seed (the 80 calibration crops count), train a throwaway
seed model, and let it pre-fill the remaining ~500 so you *review & correct*
instead of labeling from scratch. Every prediction is still reviewed by you —
the seed model is a labeling aid, discarded before final training.

**Step 4b-1 — build the subset:**
```powershell
python scripts\select_subset.py --tasks data\tasks\london_tasks.json data\tasks\ffhq_tasks.json --calibration exports\calib_a.json --target 800 --seed-size 300
```
Writes `data\tasks\subset_seed.json` (~300, incl. your calibration crops) and
`data\tasks\subset_remainder.json` (~500, heavily 40+ for grade 3-4 coverage).

**Step 4b-2 — hand-label the seed (~220 new crops):**
- New project `dermalens-seed`, same XML, same two storages, import
  `subset_seed.json`. The 80 calibration crops are already in your earlier
  project — re-labeling them here keeps one clean export, or you can merge.
- ~15–20 s/crop, sessions ≤ 45 min, `quality_reject` liberally, atlas open.
- Export JSON → `exports\seed.json`, then:
```powershell
python scripts\prepare_training_csv.py --input exports\seed.json --output data\annotations\seed.csv
python -m src.data.splits --annotations data\annotations\seed.csv --output data\seed_splits --train 0.8 --val 0.2 --test-internal 0.0 --test-external 0.0
```

**Step 4b-3 — train the seed model (Colab, ~10 min):**
Bundle and train with `configs/seed_pretrain.yaml` (see docs/COLAB.md;
use `--splits data\seed_splits` when bundling). Download `best.pt` to
`experiments\seed_pretrain\`.

**Step 4b-4 — pre-annotate the remainder:**
```powershell
python scripts\predict_preannotations.py --checkpoint experiments\seed_pretrain\best.pt --tasks data\tasks\subset_remainder.json --output data\tasks\subset_remainder_preds.json
```

**Step 4b-5 — review the pre-filled remainder (the fast part):**
- New project `dermalens-remainder`, same XML, same two storages, import
  `subset_remainder_preds.json`.
- Settings → Annotation → enable **"Use predictions to prelabel tasks"**,
  model version `seed_v1` (so each task opens pre-filled in an editable
  annotation).
- Data Manager → **sort by Prediction score ASCENDING**. Scrutinise the
  low-score (uncertain) crops and anything predicted grade 3/4; fast-accept
  the confident grade-0/1 majority with one Submit.
- *(Optional)* after ~250 reviewed, retrain on ~550 and re-run 4b-4 on the
  last ~250 for better pre-fills.
- Export JSON → `exports\remainder.json`.

**Step 4b-6 — combine to the training CSV:**
```powershell
python scripts\prepare_training_csv.py --input exports\seed.json --output data\annotations\seed.csv
python scripts\prepare_training_csv.py --input exports\remainder.json --output data\annotations\remainder.csv
```
Concatenate `seed.csv` + `remainder.csv` → `data\annotations\all_annotations.csv`
(keep one header). Subject IDs flow from the task JSON, so London
neutral+smiling share a subject_id — verify the London subject count looks
like people, not photos.

**CHECK THE GRADE HISTOGRAM NOW:** if grades 3+4 combined < 80 crops, top up
*before* splitting: `build_ffhq_subset.py --target 250 --min-age 55`, crop,
add to the remainder, pre-annotate, review. Cheaper now than after a failed
Milestone 2.

---

## Step 5 — Split the data (leak-proofing)

```powershell
python -m src.data.splits --annotations data\annotations\all_annotations.csv --output data\splits
python -m pytest tests\test_split_leakage.py -q
Get-FileHash data\splits\test_external.csv -Algorithm SHA256 >> data\splits\external_hash.txt
```

Splits BY SUBJECT into train/val/test_internal/test_external (70/10/10/10).
The London manifest subject map already keeps neutral+smiling photos of the
same person together.

**test_external.csv is now frozen.** Iterate on val; confirm milestones once
on test_internal; touch test_external exactly once, at the end.

---

## Step 6 — Train on Colab (Milestone 1, then 2)

```powershell
python scripts\make_colab_bundle.py --version v1
```

Upload the zip to Drive, open `notebooks/colab_train.ipynb`, follow
`docs/COLAB.md`. Stage order: `baseline_binary` (~10–20 min) →
`ordinal_severity` warm-started (~20–35 min) → optional `multitask`.
Sessions auto-resume from `last.pt` after disconnects.

**Interpreting binary AUROC:**
- 0.85–0.93 → healthy, iterate via error analysis below.
- ≈ 0.5 → model learned nothing; check labels and crops.
- > 0.97 → SUSPICIOUS — almost certainly subject leakage.

After each run, locally:
```powershell
python scripts\error_analysis.py --checkpoint experiments\baseline_binary\best.pt --csv data\splits\val.csv
python scripts\evaluate.py --checkpoint experiments\baseline_binary\best.pt --csv data\splits\val.csv
```

| Pattern in the error grids | Diagnosis | Fix |
|---|---|---|
| FPs all have dark shadows | Model thinks shadow = bag | More shadow-negatives; multitask dark-circles head helps |
| FNs are all grade-1 | Fuzzy 0/1 label boundary | Re-read atlas, relabel ~30 borderline crops |
| Errors cluster in dark skin tones | Dataset imbalance | Oversample those MST shades NOW |
| Big per-source QWK gap (evaluate.py) | London-studio vs FFHQ-wild domain shift | Stronger photometric augmentation |

One variable per run; log every run in `experiments/RUNLOG.md`.

**Milestone gates:** when val AUROC ≥ ~0.91 (or val QWK ≥ ~0.72), run
evaluate.py ONCE on `test_internal.csv`. Stuck below 0.85 AUROC after label
cleanup + 2 encoders? The problem is data — collect +500 crops, don't burn runs.

**If QWK plateaus < 0.70 (fallback ladder, in order):**
1. +200–400 FFHQ crops at `--min-age 55`, annotate, re-split (bundle v2).
2. Re-grade the 100 worst train-set severity errors (model as noise detector).
3. Collapse to 4 grades (merge 3+4) — product decision, discuss first.
4. Ship within-one ≥ 0.90 as the UX metric; report QWK honestly.

---

## Step 7 — Decision layer + API (local)

```powershell
$env:DERMALENS_CHECKPOINT='experiments\ordinal_severity\best.pt'
.\.venv\Scripts\uvicorn api.main:app --port 8000
# in another terminal:
curl.exe -X POST http://localhost:8000/analyze/under-eye -F "file=@selfie.jpg"
```

Try at minimum: a good selfie, a blurry photo, a dark photo, a rotated face,
a photo with no face. Verify each returns the right `decision` and message.
`MIN_MEAN_CONFIDENCE` in inference.py is a product decision, not a constant.

---

## Step 8 — Final eval, export, model card

1. Verify the frozen hash matches, then the single shot:
```powershell
python scripts\evaluate.py --checkpoint experiments\ordinal_severity\best.pt --csv data\splits\test_external.csv --tag final_external
```
The number is the number — no re-runs, no tuning afterward. An
internal-external gap > 0.05 AUROC / 0.08 QWK is a finding to report.

2. ONNX export (must print "equivalence VALIDATED"):
```powershell
python -m src.deployment.export_onnx --checkpoint experiments\ordinal_severity\best.pt --output experiments\ordinal_severity\model.onnx
```

3. Model card: fill `docs/MODEL_CARD_TEMPLATE.md` with real numbers, the
   annotator self-consistency stats, subgroup tables, CC BY attribution list,
   and the FFHQ prototype-only caveat. Honest numbers keep the next 6 months sane.

---

## When things go wrong — quick triage

| Symptom | First thing to check |
|---|---|
| `CUDA out of memory` | Lower `batch_size` in the YAML to 32 or 16 |
| Loss is NaN | Set `mixed_precision: false`; check crop_log for corrupt images |
| Val metrics frozen at 0 | All-one-class val split? Check split distributions |
| Colab disconnects mid-run | Re-run the train cell — it auto-resumes from `last.pt` |
| MediaPipe import error | venv must be Python 3.11/3.12 with `mediapipe==0.10.14` |
| Accuracy suspiciously perfect | Leakage. Rerun splits, verify subject_id correctness |
| Dataset raises FileNotFoundError | A split CSV references missing crops — fix paths, don't bypass |
