# DermaLens Eye-Bag Detection Model — Specification

**Version:** 1.0 (prototype sprint)
**Date:** Day 1
**Product:** DermaLens

---

## 1. Intended Use

> DermaLens estimates the visible appearance and severity of under-eye bags from a facial
> photograph to support skincare guidance and progress tracking. It does not diagnose medical
> conditions or determine the underlying cause of swelling.

The model answers one question: **Is visible puffiness present beneath this eye, and how severe does it appear?**

It does NOT answer:
- What is causing the swelling?
- Is this medically significant?
- Does the user need treatment?

---

## 2. Output Contract

Every inference call returns this JSON:

```json
{
  "eye_bags": {
    "left": {
      "present_probability": 0.87,
      "severity_grade": 2,
      "severity_label": "moderate",
      "confidence": 0.81
    },
    "right": {
      "present_probability": 0.79,
      "severity_grade": 2,
      "severity_label": "moderate",
      "confidence": 0.76
    }
  },
  "confounders": {
    "dark_circles_visible": true,
    "makeup_detected_or_suspected": false,
    "significant_asymmetry": false
  },
  "quality": {
    "accepted": true,
    "pose_ok": true,
    "lighting_ok": true,
    "blur_ok": true,
    "face_detected": true
  },
  "decision": "show_guidance"
}
```

### `decision` values

| Value              | Meaning                                          |
|--------------------|--------------------------------------------------|
| `show_guidance`    | Confident result, safe to show                   |
| `retake_requested` | Quality gate failed — ask user to retake photo   |
| `abstain`          | Model uncertain — do not show a severity grade   |

---

## 3. Severity Scale (Annotators MUST use exactly this rubric)

| Grade | Label           | What to look for                                                     |
|------:|-----------------|----------------------------------------------------------------------|
| 0     | Not present     | No noticeable puffiness. Under-eye area is flat.                     |
| 1     | Mild            | Slight fullness. Only visible on close inspection. Minimal shadow.   |
| 2     | Moderate        | Clearly visible soft bulge beneath the lower eyelid. Noticeable in normal lighting. |
| 3     | Pronounced      | Strong, obvious bulging. Lower eyelid visibly projects forward. Significant shadow. |
| 4     | Very pronounced | Severe, extensive puffiness. Major contour change from eyelid to cheek. |

**Annotation rules:**
- Annotate what you SEE, not what you think is causing it.
- When uncertain between two adjacent grades, pick the LOWER one.
- Grade EACH eye separately — left and right may differ.
- If image quality is too poor to grade reliably, mark as `quality_reject` and do not assign a grade.

---

## 4. Confounder Definitions

These are separate labels from severity. Annotate them independently.

| Confounder                | Label as present when...                                            |
|---------------------------|---------------------------------------------------------------------|
| `dark_circles_visible`    | Visible discoloration/pigmentation beneath the eye, separate from puffiness |
| `makeup_suspected`        | Foundation, concealer, under-eye patches, or eye makeup is visible |
| `significant_asymmetry`   | Left and right eyes differ by ≥ 2 severity grades                  |

---

## 5. What This Model Does NOT Detect

These are separate concerns that will be handled by separate models or deferred:

- **Tear-trough hollowing** (volume loss = concave, opposite of puffiness)
- **Dark circle severity** (tracked as binary confounder only in prototype)
- **Under-eye fine lines / wrinkles** (separate concern, separate model)
- **Medical causes** of swelling (never infer this)

---

## 6. Abstention Rules

The model MUST return `decision: "abstain"` when ANY of these are true:

- Face landmark extraction failed (MediaPipe returned no face)
- Mean confidence across both eyes < 0.40
- Quality gate rejected the image
- Left-right grade difference ≥ 2 → triggers asymmetry safety message

---

## 7. Safety Messaging (Use These Exact Strings in the App)

### Significant Asymmetry
```
The visible appearance is uneven between the two eye areas.
DermaLens cannot determine the cause.
Consider seeking professional advice if this is new, persistent, painful, or concerning.
```

### Abstention — Poor Quality
```
We could not analyse the under-eye area reliably.
Please retake the photo in even, natural lighting with your eyes fully open.
```

### Standard Result Footer (Always Show)
```
This is cosmetic skincare guidance, not a medical diagnosis.
```

---

## 8. Prototype Limitations (What Is NOT Done Yet)

The following are deferred to post-sprint:

| Item                                      | Reason Deferred              |
|-------------------------------------------|------------------------------|
| Multi-annotator inter-rater reliability   | Needs time + multiple annotators |
| Dermatologist rubric review               | Needs external expert        |
| External frozen test set                  | Needs new participants       |
| Fairness audit across MST skin tones      | Needs diverse dataset        |
| Mobile student distillation               | Post model validation        |
| Progress tracking validation              | Post model validation        |
| Segmentation mask head                    | Post basic severity working  |
| Production training consent flow          | Legal/product work           |

---

## 9. Architecture Summary (What We're Building)

```
Full face image
       ↓
  Quality gate (blur, pose, lighting, face size)
       ↓
  MediaPipe face landmark extraction
       ↓
  Left under-eye crop (256×160) + Right under-eye crop (256×160)
       ↓
  ConvNeXt-Tiny encoder (pretrained ImageNet)
       ↓
  Shared projection layer (512 units)
       ├── Presence head     → P(eye bag present)
       ├── Severity head     → CORAL ordinal grades 0-4
       └── Dark circles head → P(dark circles visible)
       ↓
  Calibration + asymmetry check
       ↓
  Output JSON + decision
```

---

## 10. Dataset Requirements (Prototype Sprint)

- Target: **400–500 annotated under-eye images** by end of Day 3
- Sources: existing dark_circles export + public datasets + self-captured
- Annotation tool: Label Studio
- Each image labeled: severity (0-4) per eye + dark circles present (yes/no)
- Minimum per grade: aim for at least 50 images at grade 2-3 (hard to find, but important)
