#!/usr/bin/env python3
"""
Pre-annotate Label Studio tasks with a seed-trained model.

Runs the model over a task JSON and writes the SAME tasks with a Label Studio
`predictions[]` block attached, pre-filling all five required fields so a crop
the model gets right is a single Submit. Enable "Use predictions to prelabel"
in the project's Annotation settings (model_version "seed_v1") and the
predictions are copied into an editable annotation — you confirm or correct.

`score` is the model's severity confidence, so sorting the Data Manager by
prediction score ASCENDING surfaces the crops to scrutinise first.

This is a labeling AID. Every prediction is reviewed by a human before it
enters training; the seed model is discarded afterwards.

Usage:
    python scripts/predict_preannotations.py ^
        --checkpoint experiments/seed_pretrain/best.pt ^
        --tasks data/tasks/subset_remainder.json ^
        --output data/tasks/subset_remainder_preds.json

Then re-run after a second labeling round with a retrained checkpoint to
improve the pre-fills on whatever is still unlabeled.
"""

import argparse
import sys
import uuid
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

from src.data.augmentations  import get_val_transforms
from src.models.multitask    import load_model_from_checkpoint
from src.models.ordinal_head import coral_logits_to_grade, coral_grade_confidence

# Choice strings MUST match the labeling XML in docs/annotation_guide.md exactly.
SEVERITY_LABELS = [
    "0 - Not present", "1 - Mild", "2 - Moderate",
    "3 - Pronounced", "4 - Very pronounced",
]
LS_URL_PREFIX = "/data/local-files/?d="
MODEL_VERSION = "seed_v1"


def local_path(eye_crop_url: str, data_root: Path) -> Path:
    rel = eye_crop_url.split("?d=", 1)[-1] if "?d=" in eye_crop_url else eye_crop_url
    return data_root / rel


def confidence_bucket(conf: float) -> str:
    if conf >= 0.60:
        return "high"
    if conf >= 0.35:
        return "medium"
    return "low"


def choice_result(name: str, value: str) -> dict:
    return {
        "id": uuid.uuid4().hex[:10],
        "type": "choices",
        "from_name": name,
        "to_name": "eye_crop",
        "value": {"choices": [value]},
    }


def main():
    ap = argparse.ArgumentParser(description="Attach model predictions to LS tasks")
    ap.add_argument("--checkpoint", required=True, help="Seed model best.pt")
    ap.add_argument("--tasks", required=True, help="Task JSON to pre-annotate")
    ap.add_argument("--output", required=True, help="Output task JSON with predictions")
    ap.add_argument("--data-root", default="data",
                    help="Root the LS '?d=' paths are relative to (default: data)")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(args.checkpoint, device)
    if model.severity_head is None:
        print("ERROR: checkpoint has no severity head — train the seed model with "
              "configs/seed_pretrain.yaml (use_severity_head: true).")
        sys.exit(1)
    has_dc = model.dark_circles_head is not None
    transform = get_val_transforms()
    data_root = Path(args.data_root)

    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    print(f"Pre-annotating {len(tasks)} tasks on {device} ...")

    out_tasks, missing = [], 0
    # Process in mini-batches for speed (CPU convnext over ~500 crops).
    for start in range(0, len(tasks), args.batch_size):
        chunk = tasks[start:start + args.batch_size]
        imgs, valid = [], []
        for t in chunk:
            p = local_path(t["data"]["eye_crop"], data_root)
            if not p.is_file():
                missing += 1
                valid.append(False)
                continue
            imgs.append(transform(Image.open(p).convert("RGB")))
            valid.append(True)

        preds_by_pos = {}
        if imgs:
            batch = torch.stack(imgs).to(device)
            with torch.no_grad():
                out = model(batch)
            grades = coral_logits_to_grade(out["severity_logits"].cpu())
            confs  = coral_grade_confidence(out["severity_logits"].cpu())
            dcs = (torch.sigmoid(out["dark_circles_logit"].cpu()) > 0.5) if has_dc else None
            j = 0
            for pos, ok in enumerate(valid):
                if ok:
                    preds_by_pos[pos] = (int(grades[j]), float(confs[j]),
                                         bool(dcs[j]) if has_dc else False)
                    j += 1

        for pos, t in enumerate(chunk):
            new_task = {"data": t["data"]}
            if pos in preds_by_pos:
                grade, conf, dc = preds_by_pos[pos]
                results = [
                    choice_result("quality_reject", "usable"),
                    choice_result("severity", SEVERITY_LABELS[grade]),
                    choice_result("dark_circles", "yes" if dc else "no"),
                    choice_result("makeup_suspected", "no"),
                    choice_result("annotation_confidence", confidence_bucket(conf)),
                ]
                new_task["predictions"] = [{
                    "model_version": MODEL_VERSION,
                    "score": round(conf, 4),
                    "result": results,
                }]
            out_tasks.append(new_task)

    Path(args.output).write_text(json.dumps(out_tasks, indent=2), encoding="utf-8")
    n_pred = sum(1 for t in out_tasks if "predictions" in t)
    print(f"Wrote {args.output}: {n_pred}/{len(out_tasks)} tasks pre-annotated"
          + (f" ({missing} crops missing on disk, left blank)" if missing else ""))
    print("\nIn Label Studio: new project -> add the two local storages -> import")
    print("this file. Settings -> Annotation -> enable 'Use predictions to")
    print(f"prelabel' (model_version {MODEL_VERSION}). In Data Manager, sort by")
    print("Prediction score ASCENDING and review uncertain crops first.")


if __name__ == "__main__":
    main()
