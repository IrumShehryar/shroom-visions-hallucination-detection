"""
Builds a human-readable review log for the TEST set predictions.

The test set is unlabeled -- there is no gold to score against, so this does
NOT compute IoU or any correctness metric (unlike test_offline.py, which does
that for the labeled training set). Instead it dumps everything a human needs
to judge each prediction by eye: the image path (to open and look at), the
prompt, the full response, and for each flagged span -- the exact substring,
label, confidence, and the model's own stated reason for flagging it.

Run this AFTER generate_test_predictions.py has produced
test_predictions_raw_cache.json.
"""

import json
import os
from collections import defaultdict
from datetime import datetime

RAW_CACHE_PATH = r"D:\SHROOM\test_predictions_raw_cache.json"
TEST_DATA_PATH = r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.test.en.unlabeled.jsonl"
IMAGES_DIR = r"D:\SHROOM\distrib\images\shroom-vis-images"
OUTPUT_DIR = r"D:\SHROOM\test_analysis_outputs"

from src.span_mapper import map_spans_to_characters


def load_test_rows():
    rows = {}
    with open(TEST_DATA_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def load_cache():
    with open(RAW_CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def primary_predicted_label(pred_labels):
    if not pred_labels:
        return "none_predicted"
    return pred_labels[0]["label"].lower()


def build_records():
    test_rows = load_test_rows()
    cache = load_cache()

    records = []
    for item in cache:
        sample_id = item["id"]
        row = test_rows.get(sample_id)
        if row is None:
            continue
        raw = item.get("haiku_reasoning", "[]").replace("```json", "").replace("```", "").strip()
        try:
            llm_output = json.loads(raw)
        except json.JSONDecodeError:
            llm_output = []
        pred_labels = map_spans_to_characters(llm_output, row.get("response", ""))
        records.append({
            "id": sample_id,
            "row": row,
            "llm_output": llm_output,
            "pred_labels": pred_labels,
        })
    return records


def write_sample_block(log, r):
    row = r["row"]
    sample_id = r["id"]
    response = row.get("response", "")
    image_path = os.path.join(IMAGES_DIR, row.get("image_name", ""))

    log.write(f"\n{'=' * 20} ID: {sample_id} {'=' * 20}\n")
    log.write(f"IMAGE: {image_path}\n")
    log.write(f"IMAGE EXISTS: {os.path.exists(image_path)}\n")
    log.write(f"PROMPT: {row.get('prompt', '')}\n")
    log.write(f"\n--- FULL RESPONSE ---\n{response}\n")

    if not r["pred_labels"]:
        log.write("\nPREDICTED: (no spans flagged -- pipeline judged this response clean)\n")
    else:
        log.write("\nPREDICTED SPANS:\n")
        # match each mapped span back to its raw entry (same order) to surface "reason"
        for mapped, raw_entry in zip(r["pred_labels"], r["llm_output"]):
            span_text = response[mapped["start"]:mapped["end"]]
            reason = raw_entry.get("reason", "(no reason given)")
            log.write(
                f"  [{mapped['start']}:{mapped['end']}] '{span_text}' | "
                f"{mapped['label']} | confidence: {mapped['prob']}\n"
                f"    reason: {reason}\n"
            )


def group_by_bucket(records):
    by_bucket = defaultdict(list)
    for r in records:
        bucket = primary_predicted_label(r["pred_labels"])
        by_bucket[bucket].append(r)
    return by_bucket


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    records = build_records()
    by_bucket = group_by_bucket(records)

    log_path = os.path.join(OUTPUT_DIR, "full_prediction_log.txt")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("TEST SET PREDICTION LOG (no gold labels -- for manual review only)\n")
        log.write(f"Generated: {datetime.now()}\n")
        log.write("=" * 70 + "\n")
        log.write(f"Total predictions logged: {len(records)}\n")
        log.write("Breakdown by first-flagged label (or 'none_predicted' if empty):\n")
        for bucket, items in sorted(by_bucket.items(), key=lambda kv: -len(kv[1])):
            log.write(f"  {bucket}: {len(items)}\n")
        log.write("=" * 70 + "\n")

        for bucket, items in sorted(by_bucket.items(), key=lambda kv: -len(kv[1])):
            log.write(f"\n\n{'#' * 70}\nBUCKET: {bucket.upper()} ({len(items)} samples)\n{'#' * 70}\n")
            for r in items:
                write_sample_block(log, r)

    print(f"Wrote {len(records)} prediction records to: {log_path}")
    print("\nBreakdown by first-flagged label:")
    for bucket, items in sorted(by_bucket.items(), key=lambda kv: -len(kv[1])):
        print(f"  {bucket}: {len(items)}")


if __name__ == "__main__":
    main()
