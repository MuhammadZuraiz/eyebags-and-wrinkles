# DermaLens Eye-Bag Model — Model Card (v0.1 prototype)

> Fill every ___ with real measured numbers on Day 10. No aspirational values.

## Model details
- Architecture: ConvNeXt-Tiny encoder → 512-d projection → presence (BCE) +
  severity (CORAL, grades 0–4) + dark-circles (BCE) heads
- Input: 256×160 under-eye crop, orientation-normalised (outer corner left)
- Training data: ___ crops from ___ subjects, annotated by ___ annotator(s)
- Checkpoint: experiments/multitask/best.pt (epoch ___)

## Intended use
Cosmetic guidance on visible under-eye bags inside the DermaLens app.
**Not a medical device. Not a diagnosis.** Output always carries the
disclaimer string from spec §7.

## Metrics (test_internal / test_external)
| Metric | Internal | External | Spec target |
|---|---|---|---|
| Presence AUROC | ___ | ___ | — |
| Presence sensitivity | ___ | ___ | ≥ 0.85 worst group |
| Severity QWK | ___ | ___ | ≥ 0.75 |
| Within-one-grade | ___ | ___ | ≥ 0.90 |
| ECE | ___ | ___ | no gross miscalibration |

## Fairness (subgroup_report on test sets)
| MST group | n | Sensitivity | QWK | Reliable (n≥30)? |
|---|---|---|---|---|
| ___ | | | | |

Sensitivity gap across reliable groups: ___ (target ≤ 0.05)

## Known failure modes (from error-analysis grids)
1. ___
2. ___
3. ___

## Abstention behaviour
- Mean-confidence threshold: 0.40 → abstain rate on test_internal: ___%
- Asymmetry rule (grade diff ≥ 2) triggered on ___% of test images

## Limitations
- Single annotator labels (no inter-rater agreement measured yet)
- Dataset size ___ crops — below the ~2k+ typically needed for stable grade-4 estimates
- No makeup detection in v0.1 (`makeup_detected_or_suspected` is always false)
- Trained/evaluated on ___ lighting conditions; unknown behaviour outside them
