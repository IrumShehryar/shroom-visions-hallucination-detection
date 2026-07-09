import json
import os
import re
import nltk
from datetime import datetime

# Ensure NLTK packages are loaded for any text parsing checks
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

def calculate_iou(pred_spans, gold_spans, agreement_threshold=0.5):
    """Recalculates IoU by filtering out human noise under the threshold."""
    pred_chars = set()
    gold_chars = set()
    
    for span in pred_spans:
        for i in range(span["start"], span["end"]):
            pred_chars.add(i)
    
    for span in gold_spans:
        prob = span.get("prob", 1.0)
        if prob >= agreement_threshold:
            for i in range(span["start"], span["end"]):
                gold_chars.add(i)
    
    if not pred_chars and not gold_chars:
        return 1.0
    if not pred_chars or not gold_chars:
        return 0.0
        
    return len(pred_chars & gold_chars) / len(pred_chars | gold_chars)

def run_error_analysis(
    results_path=r"D:\SHROOM\evaluation_results.json",
    originals_path=r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl",
    output_path=r"D:\SHROOM\error_analysis_log.txt"
):
    with open(results_path) as f:
        results = json.load(f)

    originals = {}
    with open(originals_path) as f:
        for line in f:
            row = json.loads(line)
            originals[row["id"]] = row

    # --- RE-EVALUATE AND UPDATE DATA WITH NEW CONSENSUS THRESHOLD ---
    updated_results = []
    for r in results:
        gold_labels = r.get("gold_labels", [])
        pred_labels = r.get("pred_labels", [])
        
        # Recalculate IoU with the 0.5 threshold filter applied
        new_iou = calculate_iou(pred_labels, gold_labels, agreement_threshold=0.5)
        
        # Build an updated tracking dictionary for this row
        updated_row = r.copy()
        updated_row["iou"] = new_iou
        
        # Filter the gold labels inside the logs too so you only print what matters
        updated_row["filtered_gold_labels"] = [
            g for g in gold_labels if g.get("prob", 1.0) >= 0.5
        ]
        updated_results.append(updated_row)

    with open(output_path, "w", encoding="utf-8") as log:
        
        def write(text=""):
            log.write(text + "\n")
            print(text)

        # Calculate tracking counts across all 50 samples using new metrics
        total_samples = len(updated_results)
        zero_iou_count = sum(1 for r in updated_results if r['iou'] == 0.0)
        perfect_iou_count = sum(1 for r in updated_results if r['iou'] == 1.0)
        partial_iou_count = sum(1 for r in updated_results if 0.0 < r['iou'] < 1.0)

        write(f"ERROR ANALYSIS LOG (CONSENSUS FILTERED AT 0.5)")
        write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        write(f"Total samples: {total_samples}")
        write(f"Perfect IoU samples (1.0): {perfect_iou_count}")
        write(f"Partial IoU samples (0.0 < IoU < 1.0): {partial_iou_count}")
        write(f"Zero IoU samples (0.0): {zero_iou_count}")
        write("="*70)

        # 1. Group the 4 distinct error states using updated consensus metrics
        false_positives = [r for r in updated_results if r["iou"] == 0.0 and len(r["pred_labels"]) > 0 and len(r["filtered_gold_labels"]) == 0]
        missed = [r for r in updated_results if r["iou"] == 0.0 and len(r["pred_labels"]) == 0 and len(r["filtered_gold_labels"]) > 0]
        wrong_location = [r for r in updated_results if r["iou"] == 0.0 and len(r["pred_labels"]) > 0 and len(r["filtered_gold_labels"]) > 0]
        partial_matches = [r for r in updated_results if 0.0 < r["iou"] < 1.0]

        # 2. Iterate through all categorized sections systematically
        for section_title, section_cases in [
            ("FALSE POSITIVES (system flagged, filtered gold is clean)", false_positives),
            ("MISSED DETECTIONS (system clean, filtered gold has errors)", missed),
            ("WRONG LOCATION (both have spans but 0.0 overlap)", wrong_location),
            ("PARTIAL MATCHES (captured correct error context, boundary alignment mismatch)", partial_matches)
        ]:
            write(f"\n{'='*70}")
            write(f"{section_title} — {len(section_cases)} cases")
            write("="*70)

            for r in section_cases:
                original = originals.get(r["id"], {})
                response = original.get("response", "")

                write(f"\nID: {r['id']}")
                write(f"PROMPT: {original.get('prompt', '')}")
                write(f"IMAGE: {original.get('image_name', '')}")
                write(f"New Filtered IoU: {r['iou']}")
                write()
                write("FULL RESPONSE:")
                write(response)
                write()
                write("GOLD LABELS (Filtered to Consensus >= 0.5):")
                for label in r.get("filtered_gold_labels", []):
                    span_text = response[label["start"]:label["end"]]
                    write(f"  [{label['start']}:{label['end']}] '{span_text}' | {label['label']} | annotator agreement: {label['prob']:.2f}")
                write()
                write("PREDICTED LABELS:")
                if r.get("pred_labels"):
                    for label in r["pred_labels"]:
                        span_text = response[label["start"]:label["end"]]
                        write(f"  [{label['start']}:{label['end']}] '{span_text}' | {label['label']} | confidence: {label['prob']:.2f}")
                else:
                    write("  (none — system returned clean)")
                write()
                write("HAIKU REASONING:")
                write(r.get("haiku_reasoning", "NOT SAVED — re-run to capture"))
                write("-"*70)

    print(f"\nLog saved to: {output_path}")

if __name__ == "__main__":
    run_error_analysis()