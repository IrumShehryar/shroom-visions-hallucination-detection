"""
Updated Official Scorer Bridge: Now scans the cache_outputs folder.
"""

import json
import glob
import os
from pathlib import Path
from src import format_checker
from src import scorer as official_scorer
from src.span_mapper import map_spans_to_characters

# Configuration
CACHE_DIR = r"D:\SHROOM\cache_outputs"
ORIGINALS_PATH = Path(r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl")
OUT_DIR = Path(r"D:\SHROOM\scoring_run")

def build_official_records(cache_data, originals):
    ref_dicts = []
    pred_dicts = []
    
    for item in cache_data:
        sample_id = item.get("id")
        if sample_id not in originals:
            continue
            
        # 1. Retrieve the original data for this sample
        original_row = originals[sample_id]
        response_text = original_row.get("response", "")
        
        # 2. Prepare Gold Labels (Clean them from the original_row)
        clean_gold = []
        for span in original_row.get("labels", []):
            clean_gold.append({
                "start": int(span["start"]),
                "end": int(span["end"]),
                "prob": float(span.get("prob", 1.0)),
                "label": str(span.get("label", "other"))
            })
            
        # 3. Add to Reference List
        ref_dicts.append({
            "id": sample_id,
            "labels": clean_gold,
            "text_len": len(response_text),
            "response": response_text,
            "raw_annots": original_row.get("raw_annots", {}) if "raw_annots" in original_row else {}
        })
        
        # 4. Build Prediction -- COMPUTE it live from haiku_reasoning,
        # same as test_offline.py does. The cache files never contain a
        # "pred_labels" key, so item.get("pred_labels", []) always silently
        # returned [] for every sample -- that was the bug.
        raw_haiku = item.get("haiku_reasoning", "[]").replace("```json", "").replace("```", "").strip()
        try:
            llm_output = json.loads(raw_haiku)
        except Exception:
            llm_output = []
        computed_pred_labels = map_spans_to_characters(llm_output, response_text)

        pred_labels = [
            {
                "start": int(span["start"]),
                "end": int(span["end"]),
                "prob": float(span.get("prob", 1.0)),
                "label": str(span.get("label", "other")),
            }
            for span in computed_pred_labels
        ]
        pred_dicts.append({"id": sample_id, "labels": pred_labels})
        
    return ref_dicts, pred_dicts

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load everything from cache_outputs
    all_cached_items = []
    search_path = os.path.join(CACHE_DIR, "cache_*.json")
    for file_path in glob.glob(search_path):
        with open(file_path, "r", encoding="utf-8") as f:
            all_cached_items.extend(json.load(f))
    
    originals = {}
    with open(ORIGINALS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            originals[row["id"]] = row

    ref_dicts, pred_dicts = build_official_records(all_cached_items, originals)

    # Save for the scorer
    pred_path = OUT_DIR / "predictions.jsonl"
    ref_path = OUT_DIR / "references.jsonl"
    
    with open(pred_path, "w", encoding="utf-8") as f:
        for row in pred_dicts: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(ref_path, "w", encoding="utf-8") as f:
        for row in ref_dicts: f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n=== RUNNING format_checker.py ===")
    errors, warnings, stats = format_checker.check_submission_files([pred_path], [ref_path], [])
    
    if errors:
        print(f"CRITICAL ERRORS FOUND: {len(errors)}")
        for e in errors[:5]: print("ERROR:", e)
    else:
        print("Format Check Passed! Running Scorer...")
        from src.scorer import load_jsonl_file_to_records
        
        # We need to reload them using the official loader to get the correct internal structure
        refs = load_jsonl_file_to_records(ref_path, is_ref=True, do_check=False)
        preds = load_jsonl_file_to_records(pred_path, is_ref=False, do_check=False)
        
        official_scorer.main(refs, preds, output_file=None, verbose=True)

if __name__ == "__main__":
    main()