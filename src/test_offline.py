"""
OFFLINE ERROR ANALYSIS — Command Center
Consolidated version: Scans cache_outputs folder and processes all categories.
"""

import os
import json
import glob
from collections import defaultdict
from datetime import datetime
from src.span_mapper import map_spans_to_characters

# --- CONFIGURATION ---
CACHE_DIR = r"D:\SHROOM\cache_outputs"
ORIGINALS_PATH = r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl"
OUTPUT_DIR = r"D:\SHROOM\analysis_outputs"

def calculate_iou(pred_spans, gold_spans):
    pred_chars = set()
    gold_chars = set()
    for span in pred_spans:
        for i in range(span["start"], span["end"]): pred_chars.add(i)
    for span in gold_spans:
        for i in range(span["start"], span["end"]): gold_chars.add(i)
    if not pred_chars and not gold_chars: return 1.0
    if not pred_chars or not gold_chars: return 0.0
    return len(pred_chars & gold_chars) / len(pred_chars | gold_chars)

def execute_complete_analysis(target_category=None):
    # 1. Load Originals
    originals = {}
    if os.path.exists(ORIGINALS_PATH):
        with open(ORIGINALS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                originals[row["id"]] = row

    # 2. Scan and Load Data from cache_outputs
    search_path = os.path.join(CACHE_DIR, "cache_*.json")
    files_to_process = glob.glob(search_path)
    
    if target_category:
        target_filename = f"cache_{target_category.lower()}.json"
        files_to_process = [f for f in files_to_process if target_filename in os.path.basename(f)]
    
    all_cached_items = []
    for file_path in files_to_process:
        print(f"Loading: {os.path.basename(file_path)}")
        with open(file_path, "r", encoding="utf-8") as f:
            all_cached_items.extend(json.load(f))
    
    # 3. Categorize and Analyze
    results_by_category = defaultdict(list)
    for cached_item in all_cached_items:
        sample_id = cached_item["id"]
        original_row = originals.get(sample_id, {})
        labels = original_row.get("labels", [])
        cat = labels[0].get("label", "other").lower() if labels else "none"
        
        # Prepare Data
        response_text = original_row.get("response", "")
        raw_haiku = cached_item.get("haiku_reasoning", "[]").replace("```json", "").replace("```", "").strip()
        try: llm_output = json.loads(raw_haiku)
        except: llm_output = []
        
        new_pred_labels = map_spans_to_characters(llm_output, response_text)
        gold_labels = cached_item.get("gold_labels", [])
        
        updated_row = cached_item.copy()
        updated_row["pred_labels"] = new_pred_labels
        updated_row["iou"] = calculate_iou(new_pred_labels, gold_labels)
        updated_row["gold_labels_unfiltered"] = gold_labels
        
        results_by_category[cat].append(updated_row)

    # 4. Write a combined JSON for view_sample.py (id, iou, gold_labels,
    # pred_labels, haiku_reasoning all together per sample)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results_flat = []
    for category, items in results_by_category.items():
        for r in items:
            all_results_flat.append({
                "id": r.get("id"),
                "iou": r.get("iou"),
                "gold_labels": r.get("gold_labels_unfiltered", []),
                "pred_labels": r.get("pred_labels", []),
                "haiku_reasoning": r.get("haiku_reasoning", ""),
            })
    combined_path = os.path.join(OUTPUT_DIR, "all_results.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results_flat, f, indent=2, ensure_ascii=False)
    print(f"Combined JSON for view_sample.py written to: {combined_path}")

    # 5. Write per-category .txt logs (unchanged)
    for category, items in results_by_category.items():
        log_path = os.path.join(OUTPUT_DIR, f"error_analysis_{category}.txt")
        with open(log_path, "w", encoding="utf-8") as log:
            n = len(items)
            ious = [r.get("iou", 0.0) for r in items]
            avg_iou = sum(ious) / n if n else 0.0
            perfect = sum(1 for x in ious if x == 1.0)
            zero = sum(1 for x in ious if x == 0.0)
            partial = n - perfect - zero

            log.write(f"OFFLINE ERROR ANALYSIS: {category.upper()}\nGenerated: {datetime.now()}\n")
            log.write("=" * 70 + "\n")
            log.write(f"SUMMARY\n")
            log.write(f"  Samples:     {n}\n")
            log.write(f"  Average IoU: {avg_iou:.4f}\n")
            log.write(f"  Perfect (1.0): {perfect}\n")
            log.write(f"  Partial:       {partial}\n")
            log.write(f"  Zero (0.0):    {zero}\n")
            log.write("=" * 70 + "\n")
            
            for r in items:
                sample_id = r.get("id", "UNKNOWN")
                iou_score = r.get("iou", 0.0)
                original_row = originals.get(sample_id, {})
                curr_text = original_row.get("response", "")
                
                # Retrieve raw haiku reasoning (cleaning up the markdown)
                haiku_raw = r.get("haiku_reasoning", "No reasoning found").replace("```json", "").replace("```", "").strip()

                log.write(f"\n{'='*20} IMAGE ID: {sample_id} {'='*20}\n")
                log.write(f"MODEL PERFORMANCE (IoU): {iou_score:.4f}\n")
                
                # New: Full context for easier debugging
                log.write(f"\n--- FULL MODEL RESPONSE ---\n{curr_text}\n")
                log.write(f"\n--- RAW HAIKU REASONING ---\n{haiku_raw}\n")
                
                log.write("\nGOLD LABELS (All Annotations):\n")
                for l in r.get("gold_labels_unfiltered", []):
                    span_text = curr_text[l.get('start', 0):l.get('end', 0)]
                    log.write(f"  [{l.get('start')}:{l.get('end')}] '{span_text}' | {l.get('label', 'N/A')} | agreement: {l.get('prob', 0.0)}\n")
                
                log.write("\nPREDICTED LABELS:\n")
                pred_labels = r.get("pred_labels", [])
                if pred_labels:
                    for l in pred_labels:
                        span_text = curr_text[l.get('start', 0):l.get('end', 0)]
                        model_conf = l.get('prob', 0.0)
                        log.write(f"  [{l.get('start')}:{l.get('end')}] '{span_text}' | {l.get('label', 'N/A')} | model confidence: {model_conf}\n")
                else:
                    log.write("  (No spans predicted)\n")

    print(f"Analysis complete. Logs saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    execute_complete_analysis()

