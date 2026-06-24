# DermaLens Annotation Guide

This guide turns a folder of under-eye crops into a training CSV for the
free-only prototype workflow.

Important: public/free source material is for prototype training only. Do not
represent public-only weights as legally cleared for commercial deployment.
Before launch, fine-tune or retrain on consented DermaLens volunteer data.

## 1. Source Policy

Use one source bucket per import batch and record it explicitly.

Recommended free seed source:
- Face Research Lab London Set: best open seed for workflow validation because
  it is high resolution and released as CC BY 4.0 with signed consent language.

Supplemental source:
- SCIN: useful only when the eye region is visible and the use case survives
  license review. It is not eye-bag-specific.

Avoid for launch weights unless separately cleared:
- FFHQ and FairFace, because they are Flickr/YFCC-derived and have license or
  resolution issues for this task.
- CFD, FACES, Bogazici, TED, Basel, CelebA, and other research/celebrity/web
  face sets unless the owner grants written commercial permission.

Every training row must include:

```text
image_path,severity,dark_circles,presence,eye,subject_id,
source_dataset,source_image_id,license_status,consent_status,
quality_reject,makeup_suspected,annotation_confidence,annotator_id
```

At minimum, `subject_id`, `source_dataset`, and `license_status` must be
non-empty before a row can enter training.

## 2. Install And Launch Label Studio

```powershell
py -3.11 -m venv .venv-ls
.\.venv-ls\Scripts\pip install label-studio

# launches with local-file serving env vars already set:
.\scripts\start_label_studio.ps1
```

Open `http://localhost:8080`, create a local account, then create your project.

**REQUIRED for images to load — register storages on EVERY project:**
the env vars alone are not enough in current Label Studio versions; the
`/data/local-files/?d=...` URLs only resolve for paths registered as a
project storage. Label Studio also refuses a path equal to the document
root itself, so register the TWO subfolders the tasks reference. For each
project, do this twice (Settings -> Cloud Storage -> Add Source Storage):

| Storage Title | Absolute local path |
|---|---|
| `crops` | `C:\Users\zurai\Desktop\Derma_Lens\Eye bags\data\crops` |
| `raw`   | `C:\Users\zurai\Desktop\Derma_Lens\Eye bags\data\raw` |

For both: Storage Type `Local files`, leave "Treat every bucket object as a
source file" OFF, and do NOT sync the storage (tasks come from the JSON
imports, not a folder scan).

If images 404 in the labeling view, a missing registration is the cause.

## 3. Import Tasks

Use one task per eye, not one task per full face.

Each task should contain:
- `face_image`: original face image for context
- `eye_crop`: the 256x160 under-eye crop to grade
- `subject_id`: stable person/source identity
- `source_image_id`: original photo/session ID
- `eye_side`: `left` or `right`
- `source_dataset`: for example `london_faces`
- `license_status`: for example `cc_by_4_0`
- `consent_status`: for example `documented_research_consent`

Example task JSON:

```json
[
  {
    "data": {
      "face_image": "/data/local-files/?d=faces/subj001.jpg",
      "eye_crop": "/data/local-files/?d=crops/subj001_left.jpg",
      "subject_id": "subj001",
      "source_image_id": "subj001",
      "eye_side": "left",
      "source_dataset": "london_faces",
      "license_status": "cc_by_4_0",
      "consent_status": "documented_research_consent"
    }
  }
]
```

## 4. Labeling Interface

Go to Settings -> Labeling Interface -> Code and paste:

Quality is asked FIRST. The grading fields (severity, dark circles, makeup,
confidence) only appear — and are only required — when the crop is `usable`.
On a rejected crop you click `reject` once and submit; severity is not forced
(forcing a grade on a crop you cannot judge would poison the data).

```xml
<View>
  <Header value="Visible under-eye bag annotation"/>
  <Text name="metadata"
        value="Subject: $subject_id | Source: $source_dataset | Image: $source_image_id | Eye: $eye_side"/>

  <View style="display: flex; gap: 16px; align-items: flex-start;">
    <View>
      <Header value="Full face context"/>
      <Image name="face_image" value="$face_image" zoom="true" maxWidth="420px"/>
    </View>
    <View>
      <Header value="Grade this under-eye crop"/>
      <Image name="eye_crop" value="$eye_crop" zoom="true" maxWidth="520px"/>
    </View>
  </View>

  <Header value="Image quality (answer this first)"/>
  <Choices name="quality_reject" toName="eye_crop" choice="single-radio" required="true">
    <Choice value="usable"/>
    <Choice value="reject - too blurry/dark/occluded to judge"/>
  </Choices>

  <!-- Everything below only shows / is required when the crop is usable -->
  <View visibleWhen="choice-selected" whenTagName="quality_reject"
        whenChoiceValue="usable">

    <Header value="Eye bag severity (puffiness or bulge only; ignore color)"/>
    <Choices name="severity" toName="eye_crop" choice="single-radio"
             required="true" requiredMessage="Pick a severity, or mark the crop reject above.">
      <Choice value="0 - Not present"/>
      <Choice value="1 - Mild"/>
      <Choice value="2 - Moderate"/>
      <Choice value="3 - Pronounced"/>
      <Choice value="4 - Very pronounced"/>
    </Choices>

    <Header value="Dark circles visible? (discoloration, regardless of puffiness)"/>
    <Choices name="dark_circles" toName="eye_crop" choice="single-radio" required="true">
      <Choice value="yes"/>
      <Choice value="no"/>
    </Choices>

    <Header value="Makeup suspected?"/>
    <Choices name="makeup_suspected" toName="eye_crop" choice="single-radio" required="true">
      <Choice value="yes"/>
      <Choice value="no"/>
      <Choice value="uncertain"/>
    </Choices>

    <Header value="Annotation confidence"/>
    <Choices name="annotation_confidence" toName="eye_crop" choice="single-radio" required="true">
      <Choice value="high"/>
      <Choice value="medium"/>
      <Choice value="low"/>
    </Choices>
  </View>
</View>
```

## 5. Severity Rubric

Severity means structural puffiness or bulge. Dark color alone is not an eye
bag; mark dark circles separately.

| Grade | Name | What you see |
|---|---|---|
| 0 | Not present | Smooth under-eye contour. Dark circles may still be present. |
| 1 | Mild | Slight fullness or shadow line visible only on close inspection. |
| 2 | Moderate | Clear, defined soft bulge beneath the lower eyelid. |
| 3 | Pronounced | Obvious bulge with strong contour or self-cast shadow. |
| 4 | Very pronounced | Large protruding bag, deep fold, or broad contour change. |

Tie-breakers:
- Torn between two grades: choose the lower grade and set confidence to low.
- Discoloration with no bulge: severity 0 and dark circles yes.
- Bulge with no discoloration: grade the bulge and mark dark circles no.
- Heavy under-eye makeup that hides contour: quality reject.
- If a crop feels unusual, compare it against the full-face context before
  labeling.

## 6. Calibration

Before labeling the full set:
- Label 50 eye crops.
- Take a break.
- Relabel the same 50 blind.
- If more than 10 differ by one grade, tighten the rubric.
- If any differ by two or more grades, review those examples before continuing.

For multi-annotator work, send disagreements and low-confidence labels to
adjudication before training.

## 7. Export And Convert

Export from Label Studio as CSV or JSON, then convert:

```bash
python scripts/prepare_training_csv.py ^
    --input  C:\path\to\labelstudio_export.csv ^
    --output data\annotations\all_annotations.csv ^
    --source-dataset london_faces ^
    --license-status cc_by_4_0 ^
    --consent-status documented_research_consent
```

If any person appears in more than one source photo, provide a subject map:

```bash
python scripts/prepare_training_csv.py ^
    --input C:\path\to\labelstudio_export.csv ^
    --output data\annotations\all_annotations.csv ^
    --subject-map data\subject_map.csv ^
    --source-dataset london_faces ^
    --license-status cc_by_4_0 ^
    --consent-status documented_research_consent
```

The converter preserves provenance fields from the export when present and
stamps them from CLI flags when absent. It refuses to write training-ready rows
without `subject_id`, `source_dataset`, and `license_status`.
