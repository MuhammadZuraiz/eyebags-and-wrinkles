# Training CSV Contract

One row is one under-eye crop for one eye.

## Required Columns

These columns must exist before `EyeBagDataset` will load the CSV:

| Column | Type | Notes |
|---|---|---|
| `image_path` | string | Path to the eye crop image. |
| `severity` | integer | Grade 0-4 for this eye crop. |
| `dark_circles` | binary | 1 if visible, else 0. |
| `subject_id` | string | Non-empty. Split unit for leakage prevention. |
| `source_dataset` | string | Non-empty. Example: `london_faces`, `dermalens_smoke`. |
| `license_status` | string | Non-empty. Example: `cc_by_4_0`, `prototype_only`. |

## Optional Columns

The loader keeps or defaults these columns:

| Column | Default | Notes |
|---|---|---|
| `presence` | derived from `severity > 0` | Binary eye-bag presence target. |
| `eye` | empty string | `left` or `right`. |
| `quality_reject` | 0 | Rows with 1 are dropped by default. |
| `source_image_id` | empty string | Original face/source photo ID. |
| `consent_status` | `unspecified` | Consent/release bucket. |
| `makeup_suspected` | 0 | Metadata/confounder, not currently a model head. |
| `annotation_confidence` | `medium` | Label confidence. |
| `annotator_id` | empty string | Annotator/audit identifier. |
| `mst_shade` | 0 | Monk Skin Tone, 0 means unknown. |
| `age_band` | empty string | Optional fairness/audit metadata. |
| `lighting` | empty string | Optional capture metadata. |

## Policy

Public/free source material is prototype-only unless separately cleared.
Before launch, train or fine-tune on consented DermaLens volunteer data.
