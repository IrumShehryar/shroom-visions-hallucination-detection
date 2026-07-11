"""
Picks WHICH samples to visually review per category, stratified across
outcome types so you're not just staring at failures. Skips anything
already in manual_review_log.csv.

Run this, then feed the printed ids one by one into:
    python -m src.view_sample <id>
"""

import json
import os
import csv
import random
from collections import defaultdict

ALL_RESULTS_PATH = r"D:\SHROOM\analysis_outputs\all_results.json"
DATA_PATH = r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl"
REVIEW_LOG_PATH = r"D:\SHROOM\manual_review_log.csv"

RANDOM_SEED = 7

# How many total to pick per category, and roughly how to split across
# outcome buckets (weak categories get more, strong ones get a light check).
PLAN = {
    "mischaracterization": {"total": 9,  "zero_no_pred": 3, "zero_wrong_pred": 3, "low_partial": 2, "high": 1},
    "other":               {"total": 9,  "zero_no_pred": 3, "zero_wrong_pred": 3, "low_partial": 2, "high": 1},
    "none":                {"total": 7,  "false_positive": 5, "correct_empty": 2},
    "invention":           {"total": 3,  "zero_no_pred": 1, "zero_wrong_pred": 1, "low_partial": 1},
    "ocr":                 {"total": 4,  "zero_no_pred": 1, "low_partial": 1, "high": 2},
    "miscounting":         {"total": 4,  "zero_no_pred": 1, "low_partial": 1, "high": 2},
}


def load_category_for_id(data_path):
    cat_by_id = {}
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            labels = row.get("labels", [])
            cat_by_id[row["id"]] = labels[0]["label"].lower() if labels else "none"
    return cat_by_id


def already_reviewed_ids():
    if not os.path.exists(REVIEW_LOG_PATH):
        return set()
    ids = set()
    with open(REVIEW_LOG_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(row["id"])
    return ids


def bucket_sample(r, category):
    iou = r.get("iou", 0.0)
    pred = r.get("pred_labels", [])

    if category == "none":
        # gold is empty; a nonzero prediction here IS the false positive
        return "false_positive" if pred else "correct_empty"

    if iou == 0.0:
        return "zero_no_pred" if not pred else "zero_wrong_pred"
    if iou < 0.5:
        return "low_partial"
    return "high"


def main():
    random.seed(RANDOM_SEED)

    with open(ALL_RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)
    cat_by_id = load_category_for_id(DATA_PATH)
    reviewed = already_reviewed_ids()

    by_cat_bucket = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r["id"] in reviewed:
            continue
        cat = cat_by_id.get(r["id"], "other")
        if cat not in PLAN:
            continue
        bucket = bucket_sample(r, cat)
        by_cat_bucket[cat][bucket].append(r["id"])

    print(f"(Skipping {len(reviewed)} already-reviewed sample(s) found in manual_review_log.csv)\n")

    grand_total = 0
    for cat, plan in PLAN.items():
        picks = []
        for bucket, n in plan.items():
            if bucket == "total":
                continue
            pool = by_cat_bucket[cat].get(bucket, [])
            random.shuffle(pool)
            picks.extend(pool[:n])

        print(f"=== {cat.upper()} ({len(picks)} picked) ===")
        for pid in picks:
            print(f"  python -m src.view_sample {pid}")
        print()
        grand_total += len(picks)

    print(f"Total samples to review: {grand_total}")


if __name__ == "__main__":
    main()