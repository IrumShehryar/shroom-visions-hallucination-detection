"""
Supplementary diagnostic analysis (NOT a replacement for the official score).

Computes IoU using only gold spans above a given annotator-agreement
threshold, at several thresholds, to quantify how much of the IoU gap is
attributable to low-consensus (disputed) gold labels vs. genuine model
error. Use this for your paper's discussion/limitations section, clearly
labeled as a supplementary analysis -- the official leaderboard score is
always computed on ALL gold spans, unfiltered.
"""

import json
import glob
import os
from collections import defaultdict

CACHE_DIR = r"D:\SHROOM\v1_cache_outputs"
DATA_PATH = r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl"

THRESHOLDS = [0.0, 0.34, 0.5, 0.67, 1.0]


def load_originals():
    originals = {}
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            originals[row["id"]] = row
    return originals


def load_all_cache():
    items = []
    for path in glob.glob(os.path.join(CACHE_DIR, "cache_*.json")):
        with open(path, encoding="utf-8") as f:
            items.extend(json.load(f))
    return items


def calculate_iou_filtered(pred_spans, gold_spans, min_agreement):
    filtered_gold = [g for g in gold_spans if g.get("prob", 1.0) >= min_agreement]

    pred_chars = {i for s in pred_spans for i in range(s["start"], s["end"])}
    gold_chars = {i for s in filtered_gold for i in range(s["start"], s["end"])}

    if not pred_chars and not gold_chars:
        return 1.0
    if not pred_chars or not gold_chars:
        return 0.0
    return len(pred_chars & gold_chars) / len(pred_chars | gold_chars)


def main():
    from src.span_mapper import map_spans_to_characters

    originals = load_originals()
    cache_data = load_all_cache()

    results_by_threshold = defaultdict(list)

    for item in cache_data:
        sample_id = item["id"]
        row = originals.get(sample_id)
        if not row:
            continue

        response_text = row.get("response", "")
        gold_labels = row.get("labels", [])

        raw = item.get("haiku_reasoning", "[]").replace("```json", "").replace("```", "").strip()
        try:
            llm_output = json.loads(raw)
        except json.JSONDecodeError:
            llm_output = []

        pred_labels = map_spans_to_characters(llm_output, response_text)

        for thresh in THRESHOLDS:
            iou = calculate_iou_filtered(pred_labels, gold_labels, thresh)
            results_by_threshold[thresh].append(iou)

    print(f"{'Threshold (min agreement)':<30}{'n samples':<12}{'Avg IoU':<10}")
    print("-" * 52)
    for thresh in THRESHOLDS:
        ious = results_by_threshold[thresh]
        avg = sum(ious) / len(ious) if ious else 0.0
        label = "ALL (official/unfiltered)" if thresh == 0.0 else f">= {thresh}"
        print(f"{label:<30}{len(ious):<12}{avg:.4f}")

    print("\nNote: only the 'ALL (official/unfiltered)' row matches the real")
    print("leaderboard scorer. The higher rows are a supplementary diagnostic")
    print("showing how much IoU improves when disputed/low-consensus gold")
    print("spans are excluded -- useful for the paper's discussion section,")
    print("not for reporting as your primary result.")


if __name__ == "__main__":
    main()