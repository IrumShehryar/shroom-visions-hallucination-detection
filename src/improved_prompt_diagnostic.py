"""
Cheap diagnostic test for a prompt change. Re-calls the API for ONLY the
samples listed below (not the full batch), saves results to a SEPARATE
diagnostic cache (never touches cache_outputs/), and prints old vs new IoU
side by side so you can judge the prompt change before committing API
budget to a full re-run.
"""

import json
import os
import glob

from src.prompt_builder import build_audit_prompt  # testing the isolated variant
from src.vision_llm import call_vision_llm, parse_llm_response
from src.span_mapper import map_spans_to_characters

DATA_PATH = r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl"
IMAGES_DIR = r"D:\SHROOM\distrib\images\shroom-vis-images"
CACHE_DIR = r"D:\SHROOM\v1_cache_outputs"  # <-- EDIT to match your exact renamed folder
DIAGNOSTIC_OUT = r"D:\SHROOM\diagnostic_run.json"

# ---------------------------------------------------------------------------
# 1) The 35 you already manually reviewed (direct before/after vs your verdicts)
# ---------------------------------------------------------------------------
ALREADY_REVIEWED_IDS = [
    "train-en-10593", "train-en-11663", "train-en-1612", "train-en-10126",
    "train-en-664", "train-en-11759", "train-en-1029", "train-en-11649",
    "train-en-1363", "train-en-1785", "train-en-2623", "train-en-2319",  # 10553 swapped out -- known positive control
    "train-en-10276", "train-en-2048", "train-en-10065", "train-en-11574",
    "train-en-10796", "train-en-10681", "train-en-9920", "train-en-422",
    "train-en-2444", "train-en-2379", "train-en-1245", "train-en-462",
    "train-en-582", "train-en-11658", "train-en-10588", "train-en-2475",
    "train-en-9810", "train-en-9627", "train-en-9846", "train-en-987",
    "train-en-10001", "train-en-530", "train-en-10153",
]

# ---------------------------------------------------------------------------
# 2) EDIT THIS: paste 15ish fresh, UNSEEN ids from none/invention/other here
#    (pick them from your error_analysis_*.txt files -- any ids not in the
#    list above)
# ---------------------------------------------------------------------------
FRESH_UNSEEN_IDS = [
    "train-en-2533",   # known positive control -- invention example test
    "train-en-11340",  # known positive control -- echo-question test
    "train-en-10311",  # known positive control -- invention example test
    "train-en-413","train-en-414","train-en-415","train-en-416","train-en-417",
     "train-en-418","train-en-419","train-en-420","train-en-421","train-en-423",    
     "train-en-424","train-en-425","train-en-426","train-en-427","train-en-428",
]


ALL_TEST_IDS = ALREADY_REVIEWED_IDS + FRESH_UNSEEN_IDS


def get_filename_hint(image_name):
    import re
    clean = re.sub(r'[\-_.]', ' ', re.sub(r'\.(jpg|jpeg|png|bmp|gif|webp)$', '', image_name.lower())).strip()
    return f"Background context only — image filename: {clean}. Only use visual evidence." if clean else "No hint"


def load_originals():
    originals = {}
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            originals[row["id"]] = row
    return originals


def load_old_pred_labels_from_cache(sample_ids):
    """Look up the OLD (current-prompt) predictions from cache_outputs, for comparison."""
    old_raw = {}
    for file_path in glob.glob(os.path.join(CACHE_DIR, "cache_*.json")):
        with open(file_path, encoding="utf-8") as f:
            for item in json.load(f):
                if item["id"] in sample_ids:
                    old_raw[item["id"]] = item.get("haiku_reasoning", "[]")
    return old_raw


def calculate_iou(pred_spans, gold_spans):
    pred_chars = {i for s in pred_spans for i in range(s["start"], s["end"])}
    gold_chars = {i for s in gold_spans for i in range(s["start"], s["end"])}
    if not pred_chars and not gold_chars:
        return 1.0
    if not pred_chars or not gold_chars:
        return 0.0
    return len(pred_chars & gold_chars) / len(pred_chars | gold_chars)


def main():
    originals = load_originals()
    old_raw = load_old_pred_labels_from_cache(ALL_TEST_IDS)

    # RESUME MODE: if a previous diagnostic_run.json exists, reuse its "new"
    # (already API-called) results instead of re-calling the API for the
    # same samples again. Only recomputes old_iou against the now-correct
    # cache, and only calls the API for ids NOT already present.
    previous_new_by_id = {}
    if os.path.exists(DIAGNOSTIC_OUT):
        with open(DIAGNOSTIC_OUT, encoding="utf-8") as f:
            for r in json.load(f):
                previous_new_by_id[r["id"]] = r

    results = []
    for i, sample_id in enumerate(ALL_TEST_IDS, 1):
        row = originals.get(sample_id)
        if not row:
            print(f"[{i}/{len(ALL_TEST_IDS)}] {sample_id}: not found in originals, skipping")
            continue

        response_text = row.get("response", "")
        gold_labels = row.get("labels", [])

        # OLD prediction (from existing cache, using OLD prompt's cached response)
        try:
            old_llm_output = json.loads(
                old_raw.get(sample_id, "[]").replace("```json", "").replace("```", "").strip()
            ) if sample_id in old_raw else []
        except json.JSONDecodeError:
            print(f"  Warning: could not parse OLD cached response for {sample_id}, treating as empty")
            old_llm_output = []
        old_pred = map_spans_to_characters(old_llm_output, response_text)
        old_iou = calculate_iou(old_pred, gold_labels)

        # NEW prediction (fresh API call with the NEW prompt)
        image_path = os.path.join(IMAGES_DIR, row.get("image_name", ""))
        if not os.path.exists(image_path):
            print(f"[{i}/{len(ALL_TEST_IDS)}] {sample_id}: image not found, skipping")
            continue

        if sample_id in previous_new_by_id:
            # Reuse the already-correct "new" result -- no API call needed.
            prev = previous_new_by_id[sample_id]
            new_raw_response = prev.get("new_haiku_reasoning", "[]")
            new_llm_output = parse_llm_response(new_raw_response)
            print(f"[{i}/{len(ALL_TEST_IDS)}] {sample_id}: reusing cached NEW result (no API call)")
        else:
            print(f"[{i}/{len(ALL_TEST_IDS)}] Calling API (NEW prompt) for {sample_id}...")
            audit_prompt = build_audit_prompt(
                prompt_text=row.get("prompt", ""),
                response_text=response_text,
                filename_hint=get_filename_hint(row.get("image_name", "")),
            )
            try:
                new_llm_output, new_raw_response = call_vision_llm(image_path, audit_prompt)
            except Exception as e:
                print(f"  API call failed: {e}")
                continue

        new_pred = map_spans_to_characters(new_llm_output, response_text)
        new_iou = calculate_iou(new_pred, gold_labels)

        results.append({
            "id": sample_id,
            "old_iou": round(old_iou, 4),
            "new_iou": round(new_iou, 4),
            "delta": round(new_iou - old_iou, 4),
            "old_pred_count": len(old_pred),
            "new_pred_count": len(new_pred),
            "gold_count": len(gold_labels),
            "new_haiku_reasoning": new_raw_response,
        })

        print(f"  old IoU: {old_iou:.4f} -> new IoU: {new_iou:.4f}  (delta: {new_iou - old_iou:+.4f})")

    with open(DIAGNOSTIC_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n = len(results)
    if n:
        avg_old = sum(r["old_iou"] for r in results) / n
        avg_new = sum(r["new_iou"] for r in results) / n
        improved = sum(1 for r in results if r["delta"] > 0)
        worsened = sum(1 for r in results if r["delta"] < 0)
        unchanged = n - improved - worsened

        print(f"\n{'='*50}")
        print(f"DIAGNOSTIC SUMMARY ({n} samples)")
        print(f"{'='*50}")
        print(f"Avg IoU: old={avg_old:.4f}  new={avg_new:.4f}  (delta: {avg_new-avg_old:+.4f})")
        print(f"Improved: {improved} | Worsened: {worsened} | Unchanged: {unchanged}")
        print(f"\nFull results saved to: {DIAGNOSTIC_OUT}")


if __name__ == "__main__":
    main()