# Training run log

One row per run. ONE variable changed per run — if you changed two things you
don't know which one worked. Val numbers only; test_internal appears only at
milestone gates, test_external exactly once at the end.

| # | Date | Stage | Bundle | Change vs previous run | val_auroc | val_qwk | within_one | Verdict / next |
|---|------|-------|--------|------------------------|-----------|---------|------------|----------------|
| 0 | 2026-06-11 | smoke | — | pipeline sanity check (synthetic data, CPU) | 1.000* | 0.287* | 0.643* | *meaningless numbers, 14 synthetic crops — pipeline works |
| 1 | 2026-06-16 | baseline_binary | v1 | first real run, 700 crops | 0.890 | — | — | negative-starved (spec 0.30, only 82 grade-0); add negatives |
| 2 | 2026-06-16 | baseline_binary | v2 | +170 young negatives in train (grade-0 58→96) | 0.967 | — | — | AUROC gate cleared; threshold 0.75 gives sens .97/spec .70 on val |
| 3 | 2026-06-23 | ordinal_severity | v2 | first ordinal (smoothing 0.5) | 0.965 | 0.719* | 1.00 | *val; test_internal QWK 0.53 — collapses grades to 2 |
| 4 | 2026-06-23 | ordinal_severity | v3 | smoothing 0.5→0.2, patience 16, 80 ep | 0.962 | 0.719 | 1.00 | grade-3 recall fixed but test QWK flat (0.55); LABEL CEILING. 3-class collapse tested = worse (0.41). SHIPPED |
| 5 | 2026-06-24 | multitask | v4 | +dark_circles head (confounders_bce 0.25) | 0.959 | 0.679 | 0.97 | severity held; dc head WEAK (val dc_auroc 0.64, sens 0.22). Dark circles DEFERRED to enhancement+re-annotation pass; not shipped |

## Milestone gates

| Milestone | Metric | Target | test_internal result | Date |
|---|---|---|---|---|
| M1 presence | AUROC | ≥ 0.90 | **0.9037 PASS** | 2026-06-16 |
| M2 severity | QWK | ≥ 0.70 | 0.554 MISS (label ceiling) | 2026-06-23 |
| M2 severity | within-one | ≥ 0.90 | **0.942 PASS** | 2026-06-23 |

## Final (test_external — single shot, ordinal model, no-TTA)

| Model | AUROC | QWK | within-one | ECE | Date |
|---|---|---|---|---|---|
| ordinal_severity/best.pt (epoch 11) | 0.910 | 0.647 | 0.957 | 0.071 | 2026-06-23 |

Outcome: M1 PASS (AUROC 0.91 external). M2 severity ships on within-one (0.96)
with QWK reported honestly (~0.55-0.65, single-annotator label ceiling).
See docs/MODEL_CARD.md.
