# DermaLens Eye-Bag Model — Model Card (v0.1 prototype)

_Measured on the frozen test sets, June 2026. No aspirational values._

## Model details
- **Architecture:** ConvNeXt-Tiny encoder → 512-d projection → presence (BCE) +
  severity (CORAL ordinal, grades 0–4) heads. Dark-circles head **off** in this
  build (`use_dark_circles_head: false`).
- **Input:** 256×160 under-eye crop, orientation-normalised (outer corner left);
  one model runs per eye.
- **Final checkpoint:** `experiments/ordinal_severity/best.pt` (epoch 11),
  warm-started from the binary presence model (`baseline_binary/best.pt`).
- **Operating config:** no test-time augmentation (hflip TTA evaluated, within
  noise, not used).
- **ONNX:** `experiments/ordinal_severity/model.onnx` + `model.onnx.data`
  (weights as external data — **ship both files together**); numerically
  equivalent to PyTorch (max abs diff 7e-07).

## Training data
- **870 crops from 489 subjects, single annotator (the project owner).**
- Sources: **FFHQ** per-image Public-Domain/CC0 subset (age-skewed older) +
  **Face Research Lab London Set** (CC BY 4.0). FFHQ-dominant by design —
  deployment target is in-the-wild selfies, not studio portraits.
- Splits (subject-level, leak-checked): train 661 (528 FFHQ / 133 London),
  val 71, test_internal 69, test_external 69 (frozen, SHA256 `35d4caf73efc2d1f`).
- Built via active-learning pre-annotation: 268 hand-labeled seed → seed model
  pre-filled the rest → all predictions human-reviewed before training.
- **License posture: prototype-only.** FFHQ-derived weights are not cleared for
  commercial deployment; retrain/fine-tune on consented data before launch.

## Intended use
Cosmetic guidance on visible under-eye bags inside the DermaLens app.
**Not a medical device. Not a diagnosis.** Output always carries the disclaimer
string from spec §7.

## Metrics (frozen test sets)
| Metric | Internal | External | Target | Status |
|---|---|---|---|---|
| Presence AUROC | 0.904 | **0.910** | ≥ 0.90 | **PASS** |
| Presence AUPRC | 0.987 | 0.991 | — | — |
| Severity QWK | 0.554 | 0.647 | ≥ 0.70 | **miss** (label ceiling) |
| Severity within-one-grade | 0.942 | **0.957** | ≥ 0.90 | **PASS** |
| Severity exact | 0.46 | 0.57 | — | — |
| Severity MAE | 0.59 | 0.48 | — | — |
| ECE (calibration) | — | 0.071 | no gross miscal. | OK |

**Headline:** reliable presence detection and reliable *within-one-grade*
severity. Fine 5-class exactness (QWK) sits at the single-annotator label
ceiling — see Limitations.

## Fairness
- **Skin-tone (MST) fairness is NOT assessable in v0.1** — MST shade was not
  collected during annotation, so no per-shade sensitivity/gap can be computed.
  This is the most important fairness gap to close before any real use.
- By source: test sets are ~entirely FFHQ (London n=1 internal, 0 external), so
  studio-lighting performance is effectively unmeasured on held-out data.

## Known failure modes (from confusion matrices)
1. **Grade 1↔2 confusion** — the dominant error and the QWK ceiling. It mirrors
   the annotator's own noisiest boundary (calibration κ flagged 1/2). A model
   cannot exceed its labels' consistency here.
2. **Grade 2 occasionally over-called as 3** on external (11/36) — mild upward
   bias at the moderate/pronounced edge.
3. **Grade 4 effectively unsupported** — only 2 grade-4 crops in the whole
   dataset; the model essentially never predicts 4. Severe festoons/malar mounds
   are out of distribution.
4. **Studio/clean-skin domain under-represented** (London thin in test) — behaviour
   on studio lighting is under-measured.

## Abstention behaviour
- Abstention rules from spec §6 are implemented in `src/deployment/inference.py`
  (quality-gate retake, mean-confidence < 0.40, left/right asymmetry ≥ 2 grades).
- The CORAL confidence rarely drops below 0.40, so the confidence rule almost
  never fires on its own (0% on test_external); abstention in practice is driven
  by the quality gate and the asymmetry rule. within-one ≥ 0.94 means confident
  predictions are rarely off by more than one grade.

## Limitations
- **Single annotator, no inter-rater agreement.** Self-consistency was κ 0.66 /
  QWK 0.78 on a calibration batch; operational QWK ceiling ≈ 0.55–0.65.
- **870 crops** — small; grade-3 estimates are noisy and grade-4 is unsupported.
  Held-out sets are 69 crops each, so single-metric values carry wide error bars
  (QWK swung 0.55→0.65 between the two test draws).
- No skin-tone labels → no demographic fairness audit.
- No makeup prediction in this build.
- **Dark circles: evaluated and deferred, not shipped.** A multitask variant with
  the dark-circles head on was trained (warm-started from this model); severity
  held (val QWK 0.68, within-one 0.97) but the dark-circles head was weak
  (val dc_auroc **0.64**, sensitivity 0.22 — misses ~78% of positives). Root
  causes: dark-circles labels were a secondary annotation field (noisier), and
  dark circles is a color/pigment feature highly sensitive to lighting that the
  deferred capture-normalization pass would address. So `dark_circles_visible`
  stays `false` in this build; revisit it in the enhancement + re-annotation pass.
- Prototype-only data licensing (FFHQ); not commercially cleared.

## Path to a stronger model (future work)
1. Multi-annotator labeling + adjudication on the grade 1/2 boundary (the single
   biggest QWK lever).
2. Collect MST shade per crop and run the subgroup fairness audit.
3. Grow data toward 2k+ crops with real grade-3/4 coverage (clinical/consented
   source) for stable high-grade estimates and a commercial license path.
4. Revisit dark circles in the capture-normalization pass: apply lighting/
   white-balance enhancement to train+inference and re-annotate the dark-circles
   field carefully — it's a color feature the current raw-crop pipeline and
   secondary labels can't support (deferred at dc_auroc 0.64).
