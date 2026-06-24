#!/usr/bin/env python3
"""
Select a stratified annotation subset and split it into seed + remainder.

WHY: labeling every crop solo is the bottleneck. We (a) trim the label target
to a stratified ~800 that keeps all the older faces (the grade 3-4 coverage we
cannot afford to lose), and (b) carve off a ~300-crop seed to hand-label first.
A model trained on the seed then pre-annotates the remainder (see
scripts/predict_preannotations.py) so the rest is review-and-correct, not
label-from-scratch.

Reads the task JSONs already produced by generate_ls_tasks.py (each task dict
carries data.subject_id / data.age_band / data.source_dataset / data.eye_crop),
so no re-cropping. Selection is subject-aware: all crops of one subject land in
the same bucket, and already-labeled calibration crops are forced into the seed
so they count as free seed labels.

Usage:
    python scripts/select_subset.py ^
        --tasks data/tasks/london_tasks.json data/tasks/ffhq_tasks.json ^
        --calibration exports/calib_a.json ^
        --target 800 --seed-size 300

Output:
    data/tasks/subset_seed.json       (~300 crops, hand-label these first)
    data/tasks/subset_remainder.json  (~500 crops, model pre-annotates these)
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

OLDER_BANDS = {"40_49", "50_59", "60_69", "70_79"}


def task_key(task: dict) -> str:
    return task["data"]["eye_crop"]


def load_tasks(paths) -> list:
    tasks, seen = [], set()
    for p in paths:
        for t in json.loads(Path(p).read_text(encoding="utf-8")):
            k = task_key(t)
            if k not in seen:            # de-dup across files
                seen.add(k)
                tasks.append(t)
    return tasks


def load_calibration_keys(path) -> set:
    if not path:
        return set()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {task_key(t) for t in data if "eye_crop" in t.get("data", {})}


def main():
    ap = argparse.ArgumentParser(description="Select stratified annotation subset")
    ap.add_argument("--tasks", nargs="+", required=True,
                    help="Task JSON files from generate_ls_tasks.py")
    ap.add_argument("--calibration", default="",
                    help="A calibration export (calib_a.json) — its crops are forced into the seed")
    ap.add_argument("--target", type=int, default=800, help="Total crops to label")
    ap.add_argument("--seed-size", type=int, default=300, help="Seed crops to hand-label first")
    ap.add_argument("--out-dir", default="data/tasks")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tasks = load_tasks(args.tasks)
    calib_keys = load_calibration_keys(args.calibration)
    by_key = {task_key(t): t for t in tasks}
    print(f"Loaded {len(tasks)} tasks; {len(calib_keys)} calibration crops to force into seed")

    # ── Stratified selection to --target ──────────────────────────────────
    older  = [t for t in tasks if t["data"].get("age_band") in OLDER_BANDS]
    younger = [t for t in tasks if t["data"].get("age_band") not in OLDER_BANDS]

    selected = {task_key(t) for t in older}              # keep ALL older faces
    selected |= calib_keys                               # never drop labeled crops

    # Fill the rest with younger crops, alternating sources so both the studio
    # (London) and in-the-wild (FFHQ) domains stay represented among negatives.
    young_by_src = defaultdict(list)
    for t in younger:
        if task_key(t) not in selected:
            young_by_src[t["data"].get("source_dataset", "?")].append(t)
    for lst in young_by_src.values():
        rng.shuffle(lst)
    sources = sorted(young_by_src)
    i = 0
    while len(selected) < args.target and any(young_by_src.values()):
        src = sources[i % len(sources)]
        if young_by_src[src]:
            selected.add(task_key(young_by_src[src].pop()))
        i += 1

    subset = [by_key[k] for k in selected]
    print(f"Selected {len(subset)} crops (target {args.target})")

    # ── Subject-aware seed / remainder split ──────────────────────────────
    subj_crops = defaultdict(list)
    for t in subset:
        subj_crops[t["data"].get("subject_id", task_key(t))].append(t)

    calib_subjects = {by_key[k]["data"].get("subject_id") for k in calib_keys if k in by_key}
    other_subjects = [s for s in subj_crops if s not in calib_subjects]
    rng.shuffle(other_subjects)

    seed_tasks, seed_count = [], 0
    # Calibration subjects first (already labeled), then fill to --seed-size.
    for s in sorted(calib_subjects):
        seed_tasks += subj_crops[s]
        seed_count += len(subj_crops[s])
    for s in other_subjects:
        if seed_count >= args.seed_size:
            break
        seed_tasks += subj_crops[s]
        seed_count += len(subj_crops[s])

    seed_keys = {task_key(t) for t in seed_tasks}
    remainder_tasks = [t for t in subset if task_key(t) not in seed_keys]

    # ── Write + summarise ─────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "subset_seed.json").write_text(
        json.dumps(seed_tasks, indent=2), encoding="utf-8")
    (out_dir / "subset_remainder.json").write_text(
        json.dumps(remainder_tasks, indent=2), encoding="utf-8")

    def summary(name, ts):
        src = Counter(t["data"].get("source_dataset", "?") for t in ts)
        age = Counter(t["data"].get("age_band") or "unknown" for t in ts)
        subj = len({t["data"].get("subject_id") for t in ts})
        older_n = sum(1 for t in ts if t["data"].get("age_band") in OLDER_BANDS)
        print(f"\n{name}: {len(ts)} crops, {subj} subjects, {older_n} aged 40+")
        print(f"  source: {dict(src)}")
        print(f"  age_band: {dict(sorted(age.items()))}")

    summary("SEED (hand-label first)", seed_tasks)
    summary("REMAINDER (model pre-annotates)", remainder_tasks)
    print(f"\nWrote {out_dir/'subset_seed.json'} and {out_dir/'subset_remainder.json'}")
    print("\nNext: annotate subset_seed.json in Label Studio, export, then")
    print("train the seed model and run scripts/predict_preannotations.py.")


if __name__ == "__main__":
    main()
