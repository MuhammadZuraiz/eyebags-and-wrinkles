# DermaLens Severity Grade Atlas

Keep this open in a second window during ALL annotation sessions. Grade by
comparing against these exemplars, not from memory — memory drift between
sessions is the main source of single-annotator label noise.

## How this atlas gets built

1. Annotate the 80-crop calibration batch twice (`data/tasks/calibration_tasks.json`).
2. Run `python scripts/annotation_qa.py --pass-a <export A> --pass-b <export B>`.
3. From the crops where BOTH passes agreed, pick 3–4 per grade and paste them
   below (drag the crop files into this doc folder or reference their paths).
4. Re-read the tiebreak rules whenever you return after a break.

## Grade definitions (from the spec — puffiness/bulge ONLY, ignore color)

| Grade | Name | What you see |
|---|---|---|
| 0 | Not present | Smooth under-eye contour. Dark circles may still be present. |
| 1 | Mild | Slight fullness or shadow line visible only on close inspection. |
| 2 | Moderate | Clear, defined soft bulge beneath the lower eyelid. |
| 3 | Pronounced | Obvious bulge with strong contour or self-cast shadow. |
| 4 | Very pronounced | Large protruding bag, deep fold, or broad contour change. |

## Exemplars

### Grade 0 — Not present
<!-- After calibration: ![g0-a](atlas/g0_a.jpg) ![g0-b](atlas/g0_b.jpg) ![g0-c](atlas/g0_c.jpg) -->
*(fill after calibration — crops where both passes agreed on 0)*

### Grade 1 — Mild
*(fill after calibration)*

### Grade 2 — Moderate
*(fill after calibration)*

### Grade 3 — Pronounced
*(fill after calibration)*

### Grade 4 — Very pronounced
*(fill after calibration — these will be scarce; if fewer than 3 exist in the
calibration batch, pull candidates from the FFHQ age-55+ pool)*

## Tiebreak rules

Start with the spec rules; ADD your own after `annotation_qa.py` shows which
boundary is personally noisy (usually 0/1 and 2/3):

- Torn between two grades: choose the LOWER grade and set confidence to low.
- Discoloration with no bulge: severity 0 + dark circles yes.
- Bulge with no discoloration: grade the bulge + dark circles no.
- Heavy under-eye makeup that hides contour: quality reject.
- 0 vs 1: if you need to zoom past 100% to convince yourself there is
  fullness, it is grade 0.
- 2 vs 3: grade 3 requires a self-cast shadow or a contour change visible at
  arm's-length viewing distance; a clear bulge without either stays 2.

### Calibration findings (pass A vs B, 2026: kappa 0.66, QWK 0.78)

Your disagreements were entirely at two boundaries. Apply these to tighten them:

- **0 vs 1 (7 disagreements):** Grade 1 needs a *visible* feature — a faint
  crease line OR a slight raised fullness that you can point to. "The skin
  looks a bit tired" with nothing you can point to = grade 0. When genuinely
  torn, default to 0 (the lower grade) + confidence low.
- **1 vs 2 (7 disagreements):** The test is a *defined boundary*. Grade 2 has
  a pouch with an edge you could trace; grade 1 is fullness that fades out
  with no clear lower border. If you cannot trace where the bag ends, it is 1.
- You had **no 2+ grade disagreements** — trust your high/low instinct; only
  the adjacent-grade calls need these rules.
- Grades 3-4 were nearly absent in calibration, so your consistency there is
  untested. When you hit a grade-3/4 candidate in the full set, slow down and
  compare against the atlas exemplars deliberately.
