# Training on Google Colab (free T4 GPU)

The laptop (Intel Arc iGPU, no CUDA) handles preprocessing, annotation, splits,
error analysis and the API demo. All training runs happen on Colab.

## One-time setup

1. Build the data+code bundle locally (after splits exist):

   ```powershell
   .\.venv\Scripts\python.exe scripts\make_colab_bundle.py --version v1
   ```

2. Upload `dermalens_bundle_v1.zip` to your Google Drive **root** (~60–120 MB).
3. Open `notebooks/colab_train.ipynb` in Colab (upload it or open from Drive).
4. Runtime → Change runtime type → **T4 GPU**.

## Per-session flow

Run the notebook cells top to bottom. The notebook:

- unzips the bundle into `/content/dermalens`
- restores `experiments/` from `Drive/dermalens/` so resume works across sessions
- installs the only two missing deps (`albumentations`, `pyyaml` — the training
  import chain needs no mediapipe/opencv)
- smoke-trains 2 epochs on the bundled synthetic data as a sanity check
- trains the stage selected by the `STAGE` variable, auto-resuming from
  `last.pt` if the session died mid-run, or warm-starting from the previous
  stage's `best.pt`
- backs up `experiments/` to Drive and downloads a zip for local error analysis

## Stage order (the curriculum)

| Stage | Config | Warm-start | Monitor | Expected T4 time |
|---|---|---|---|---|
| 0 (optional) | `seed_pretrain.yaml` | — | val_qwk | ~5–10 min |
| 1 | `baseline_binary.yaml` | — | val_auroc | ~10–20 min |
| 2 | `ordinal_severity.yaml` | stage 1 best.pt | val_qwk | ~20–35 min |
| 3 (optional) | `multitask.yaml` | stage 2 best.pt | val_qwk | ~25–40 min |

**Stage 0 (seed pre-annotator)** is the throwaway model that powers the
annotation-acceleration loop (WALKTHROUGH Step 4b). Bundle it with the seed
split and set `STAGE = 'seed_pretrain'`:

```powershell
.\.venv\Scripts\python.exe scripts\make_colab_bundle.py --version seed --splits data\seed_splits
```

Download its `best.pt` to `experiments\seed_pretrain\` and run
`scripts\predict_preannotations.py` locally. It is not part of the milestone
curriculum — discard it once the remainder is reviewed.

Milestone gates (evaluate ONCE on `test_internal.csv` when val looks good):

- **Milestone 1:** presence AUROC ≥ 0.90
- **Milestone 2:** severity QWK ≥ 0.70 AND within-one ≥ 0.90

`test_external.csv` is frozen — touch it exactly once, for the final report.

## After each run (locally)

```powershell
# unzip the downloaded run into experiments/, then:
.\.venv\Scripts\python.exe scripts\error_analysis.py --checkpoint experiments\baseline_binary\best.pt --csv data\splits\val.csv --out experiments\baseline_binary\error_analysis
```

Log every run (config delta + val metric) in `experiments/RUNLOG.md` —
one variable per run.

## When data changes

Re-annotation or re-splits → rebuild and upload a new bundle version
(`--version v2`) and update `BUNDLE_VERSION` in the notebook. Never mix
checkpoints across bundle versions when comparing runs.
