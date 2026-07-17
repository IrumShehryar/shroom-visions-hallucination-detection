"""
Fresh-sample validation run for the mischaracterization/invention confidence
thresholds found in src/confidence_threshold_sweep.py.

The 286-sample scoring_run and 344-sample old_prompt_v1_scoring_run sets
were both already used to discover/sanity-check those thresholds -- neither
is a clean held-out check under the CURRENT prompt. This script:

1. Picks a fresh batch of train-en rows that appear in NEITHER of those two
   sets (real API calls, real cost -- confirms before spending).
2. Caches the raw current-prompt Haiku output per sample (resumable).
3. Writes a full per-sample log + bucketed error-analysis doc, matching the
   existing analysis_outputs/ format, into validation_outputs/.
4. Scores baseline vs. the candidate threshold filters against this
   never-before-seen-by-the-filter set using the official scorer.
"""
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime

from src.prompt_builder import build_audit_prompt
from src.vision_llm import call_vision_llm
from src.span_mapper import map_spans_to_characters
from src import scorer as official_scorer

FULL_DATA_PATH = r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl"
IMAGES_DIR = r"D:\SHROOM\distrib\images\shroom-vis-images"
SCORING_RUN_PREDS = r"D:\SHROOM\scoring_run\predictions.jsonl"
OLD_PROMPT_PREDS = r"D:\SHROOM\old_prompt_v1_scoring_run\predictions.jsonl"

RAW_CACHE_PATH = r"D:\SHROOM\fresh_validation_raw_cache.json"
OUTPUT_DIR = r"D:\SHROOM\validation_outputs"

FRESH_COUNT = 40
RANDOM_SEED = 2026


def get_filename_hint(image_name):
    import re
    clean = re.sub(r'[\-_.]', ' ', re.sub(r'\.(jpg|jpeg|png|bmp|gif|webp)$', '', image_name.lower())).strip()
    return f"Background context only — image filename: {clean}. Only use visual evidence." if clean else "No hint"


def load_jsonl_ids(path):
    return {json.loads(l)["id"] for l in open(path, encoding="utf-8")}


def load_full_dataset(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_raw_cache():
    if os.path.exists(RAW_CACHE_PATH):
        with open(RAW_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_raw_cache(cache_data):
    with open(RAW_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)


def pick_fresh_rows():
    all_rows = load_full_dataset(FULL_DATA_PATH)
    excluded = load_jsonl_ids(SCORING_RUN_PREDS) | load_jsonl_ids(OLD_PROMPT_PREDS)
    pool = [r for r in all_rows if r["id"] not in excluded]
    random.seed(RANDOM_SEED)
    random.shuffle(pool)
    return pool[:FRESH_COUNT]


def run_api_calls(rows):
    cache_data = load_raw_cache()
    done_ids = {item["id"] for item in cache_data}
    remaining = [r for r in rows if r["id"] not in done_ids]
    print(f"Fresh batch: {len(rows)} samples, {len(done_ids)} already cached, {len(remaining)} to call.")

    if remaining:
        if "-y" in sys.argv or "--yes" in sys.argv:
            print(f"Proceeding with {len(remaining)} API calls (--yes flag, non-interactive).")
        else:
            confirm = input(f"Proceed with {len(remaining)} API calls? (y/n): ").strip().lower()
            if confirm != "y":
                print("Aborted.")
                return None

    for i, row in enumerate(remaining, 1):
        sample_id = row["id"]
        print(f"[{i}/{len(remaining)}] Calling API for {sample_id}...")
        image_path = os.path.join(IMAGES_DIR, row.get("image_name", ""))
        audit_prompt = build_audit_prompt(
            row.get("prompt", ""), row.get("response", ""), get_filename_hint(row.get("image_name", ""))
        )
        try:
            _, raw_response = call_vision_llm(image_path, audit_prompt)
            cache_data.append({
                "id": sample_id,
                "gold_labels": row.get("labels", []),
                "haiku_reasoning": raw_response,
            })
        except Exception as e:
            print(f"  Error on {sample_id}: {e}")
            continue

        if i % 10 == 0:
            save_raw_cache(cache_data)

    save_raw_cache(cache_data)
    print(f"Cached {len(cache_data)} total samples to {RAW_CACHE_PATH}")
    return cache_data


def calculate_iou(pred_spans, gold_spans):
    pred_chars = {i for s in pred_spans for i in range(s["start"], s["end"])}
    gold_chars = {i for s in gold_spans for i in range(s["start"], s["end"])}
    if not pred_chars and not gold_chars:
        return 1.0
    if not pred_chars or not gold_chars:
        return 0.0
    return len(pred_chars & gold_chars) / len(pred_chars | gold_chars)


def build_results(cache_data, originals):
    results = []
    for item in cache_data:
        sample_id = item["id"]
        row = originals.get(sample_id, {})
        response_text = row.get("response", "")
        raw = item.get("haiku_reasoning", "[]").replace("```json", "").replace("```", "").strip()
        try:
            llm_output = json.loads(raw)
        except json.JSONDecodeError:
            llm_output = []
        pred_labels = map_spans_to_characters(llm_output, response_text)
        gold_labels = item.get("gold_labels", [])
        results.append({
            "id": sample_id,
            "response": response_text,
            "haiku_reasoning": raw,
            "gold_labels": gold_labels,
            "pred_labels": pred_labels,
            "iou": calculate_iou(pred_labels, gold_labels),
        })
    return results


def primary_category(labels):
    return labels[0]["label"].lower() if labels else "none"


def write_full_log(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "fresh_validation_log.txt")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"FRESH VALIDATION LOG (current prompt, never used to tune thresholds)\n")
        log.write(f"Generated: {datetime.now()}\n")
        log.write(f"Samples: {len(results)}  |  Seed: {RANDOM_SEED}  |  Source: {FULL_DATA_PATH}\n")
        log.write("=" * 70 + "\n")
        for r in results:
            log.write(f"\n{'=' * 20} ID: {r['id']} {'=' * 20}\n")
            log.write(f"IoU: {r['iou']:.4f}\n")
            log.write(f"\n--- FULL RESPONSE ---\n{r['response']}\n")
            log.write(f"\n--- RAW HAIKU REASONING ---\n{r['haiku_reasoning']}\n")
            log.write("\nGOLD LABELS:\n")
            for l in r["gold_labels"]:
                span_text = r["response"][l.get("start", 0):l.get("end", 0)]
                log.write(f"  [{l.get('start')}:{l.get('end')}] '{span_text}' | {l.get('label')} | agreement: {l.get('prob')}\n")
            log.write("\nPREDICTED LABELS:\n")
            if r["pred_labels"]:
                for l in r["pred_labels"]:
                    span_text = r["response"][l["start"]:l["end"]]
                    log.write(f"  [{l['start']}:{l['end']}] '{span_text}' | {l['label']} | confidence: {l['prob']}\n")
            else:
                log.write("  (No spans predicted)\n")
    print(f"Wrote full log: {log_path}")


def write_error_analysis(results):
    by_cat = defaultdict(list)
    for r in results:
        by_cat[primary_category(r["gold_labels"])].append(r)

    for cat, items in by_cat.items():
        log_path = os.path.join(OUTPUT_DIR, f"fresh_validation_error_analysis_{cat}.txt")
        n = len(items)
        ious = [r["iou"] for r in items]
        avg_iou = sum(ious) / n if n else 0.0
        perfect = sum(1 for x in ious if x == 1.0)
        zero = sum(1 for x in ious if x == 0.0)
        partial = n - perfect - zero
        with open(log_path, "w", encoding="utf-8") as log:
            log.write(f"FRESH VALIDATION ERROR ANALYSIS: {cat.upper()}\nGenerated: {datetime.now()}\n")
            log.write("=" * 70 + "\n")
            log.write(f"SUMMARY\n  Samples: {n}\n  Average IoU: {avg_iou:.4f}\n")
            log.write(f"  Perfect (1.0): {perfect}\n  Partial: {partial}\n  Zero (0.0): {zero}\n")
            log.write("=" * 70 + "\n")
            for r in items:
                log.write(f"\n{'=' * 20} ID: {r['id']} {'=' * 20}\n")
                log.write(f"IoU: {r['iou']:.4f}\n")
                log.write(f"\n--- RAW HAIKU REASONING ---\n{r['haiku_reasoning']}\n")
    print(f"Wrote {len(by_cat)} bucketed error-analysis files to {OUTPUT_DIR}")


def filter_by_label_thresh(pred_dicts, thresholds, default=0.0):
    out = []
    for row in pred_dicts:
        labs = [l for l in row["labels"] if l["prob"] >= thresholds.get(l["label"], default)]
        out.append({"id": row["id"], "labels": labs})
    return out


def score_set(ref_dicts, pred_dicts):
    ref_sorted = sorted(ref_dicts, key=lambda d: d["id"])
    pred_sorted = sorted(pred_dicts, key=lambda d: d["id"])
    assert [d["id"] for d in ref_sorted] == [d["id"] for d in pred_sorted]
    cors = [official_scorer.score_cor(r, p) for r, p in zip(ref_sorted, pred_sorted)]
    cors_lbl = [official_scorer.score_cor_lbl(r, p) for r, p in zip(ref_sorted, pred_sorted)]
    ious = [official_scorer.score_iou(r, p) for r, p in zip(ref_sorted, pred_sorted)]
    n = len(cors)
    return sum(cors) / n, sum(cors_lbl) / n, sum(ious) / n


def run_threshold_check(results):
    ref_dicts = [{"id": r["id"], "labels": r["gold_labels"], "text_len": len(r["response"])} for r in results]
    pred_dicts = [{"id": r["id"], "labels": r["pred_labels"]} for r in results]

    configs = [
        ("baseline (no filter)", {}),
        ("mischar>=0.92", {"mischaracterization": 0.92}),
        ("mischar>=0.92 + invention>=0.5", {"mischaracterization": 0.92, "invention": 0.5}),
        ("mischar>=0.95 + invention>=0.5", {"mischaracterization": 0.95, "invention": 0.5}),
    ]

    lines = [f"FRESH VALIDATION THRESHOLD CHECK (n={len(results)}, current prompt, held out from tuning)",
             f"Generated: {datetime.now()}", "=" * 70,
             f"{'config':<40}{'Cor':<10}{'Cor+Lbl':<10}{'IoU':<10}"]
    print(lines[-1])
    for name, thresholds in configs:
        filtered = filter_by_label_thresh(pred_dicts, thresholds)
        cor, cl, iou = score_set(ref_dicts, filtered)
        line = f"{name:<40}{cor:<10.4f}{cl:<10.4f}{iou:<10.4f}"
        print(line)
        lines.append(line)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "fresh_validation_threshold_check.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote threshold check: {out_path}")


def main():
    rows = pick_fresh_rows()
    print(f"Picked {len(rows)} fresh sample ids (seed={RANDOM_SEED}), none overlapping scoring_run or old_prompt_v1_scoring_run.")

    cache_data = run_api_calls(rows)
    if cache_data is None:
        return

    originals = {r["id"]: r for r in load_full_dataset(FULL_DATA_PATH)}
    results = build_results(cache_data, originals)

    write_full_log(results)
    write_error_analysis(results)
    run_threshold_check(results)


if __name__ == "__main__":
    main()
