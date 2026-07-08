"""
OFFLINE ERROR ANALYSIS — Command Center (Updated for Official Scorer Alignment)
This file is your sole source of truth for mapping logic and diagnostic analysis.
"""
print("DEBUG: File is running!")
import json
import os
import re
import nltk
from datetime import datetime

# NLTK setup
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand",
}

def normalize_text(text):
    if not text: return ""
    return text.replace('\u2019', "'").replace('\u2018', "'").replace('\u201c', '"').replace('\u201d', '"')

def _tokenize_words(text):
    return re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", text)

def isolate_span(span_text, label, response_text):
    span_text = span_text.strip()
    label = (label or "other").lower()
    if span_text and span_text in response_text:
        words = span_text.split()
        if len(words) == 1 or label not in ("miscounting", "mischaracterization"):
            return span_text
    words = _tokenize_words(span_text)
    if not words: return span_text
    if label == "miscounting":
        for w in words:
            if w.isdigit() or w.lower() in NUMBER_WORDS: return w
        return span_text
    if label == "mischaracterization":
        try: tagged = nltk.pos_tag(words)
        except Exception: tagged = [(w, "UNK") for w in words]
        for w, tag in tagged:
            if tag in ("JJ", "JJR", "JJS"): return w
        return span_text
    return span_text

def map_spans_to_characters(llm_output, response_text):
    mapped = []
    already_used = []
    normalized_response = normalize_text(response_text)
    for item in llm_output:
        raw_span = normalize_text(item.get("span_text", "")).strip()
        label = item.get("label", "other")
        prob = item.get("prob", 0.5)
        if not raw_span: continue
        target = isolate_span(raw_span, label, normalized_response)
        if not target: continue
        idx = normalized_response.find(target)
        matched_text = target
        if idx == -1:
            idx = normalized_response.lower().find(target.lower())
        if idx == -1: continue
        end = idx + len(matched_text)
        overlaps = any(not (end <= used_start or idx >= used_end) for used_start, used_end in already_used)
        if overlaps: continue
        already_used.append((idx, end))
        mapped.append({"start": idx, "end": end, "label": label, "prob": round(float(prob), 4)})
    mapped.sort(key=lambda x: x["start"])
    return mapped

def calculate_iou(pred_spans, gold_spans, agreement_threshold=0.5):
    pred_chars = set()
    gold_chars = set()
    for span in pred_spans:
        for i in range(span["start"], span["end"]): pred_chars.add(i)
    # Using raw gold_spans to match official scorer behavior
    for span in gold_spans:
        for i in range(span["start"], span["end"]): gold_chars.add(i)
    if not pred_chars and not gold_chars: return 1.0
    if not pred_chars or not gold_chars: return 0.0
    return len(pred_chars & gold_chars) / len(pred_chars | gold_chars)

def execute_complete_analysis(
    cache_path=r"D:\SHROOM\v7_simulated_evaluation_results.json",
    originals_path=r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl",
    output_log_path=r"D:\SHROOM\error_analysis_log.txt",
):
   
    
    TARGET_IDS = [] 
    try:
        print("DEBUG: Attempting to open file...")
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
            print(f"DEBUG: cache_data contains {len(cache_data)} items.")
    except Exception as e:
        print(f"DEBUG: FAILED TO LOAD FILE. Error: {e}")
        return
    if TARGET_IDS:
        cache_data = [r for r in cache_data if r.get("id") in TARGET_IDS]
    originals = {}
    if os.path.exists(originals_path):
        with open(originals_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                originals[row["id"]] = row
    updated_results = []
    for cached_item in cache_data:
       
        sample_id = cached_item["id"]
        original_row = originals.get(sample_id, {})
        response_text = original_row.get("response", "")
        raw_haiku = cached_item.get("haiku_reasoning", "[]").replace("```json", "").replace("```", "").strip()
        try: llm_output = json.loads(raw_haiku)
        except: llm_output = []
        new_pred_labels = map_spans_to_characters(llm_output, response_text)
        gold_labels = cached_item.get("gold_labels", [])
        new_iou = calculate_iou(new_pred_labels, gold_labels)
        updated_row = cached_item.copy()
        updated_row["pred_labels"] = new_pred_labels
        updated_row["iou"] = new_iou
        # REMOVED PROBABILITY FILTER: Now passing all gold labels
        updated_row["gold_labels_unfiltered"] = gold_labels 
        updated_results.append(updated_row)

    os.makedirs(os.path.dirname(output_log_path), exist_ok=True)
   # ... (inside execute_complete_analysis)

    with open(output_log_path, "w", encoding="utf-8") as log:
        def write(text=""): log.write(text + "\n")
        write("OFFLINE ERROR ANALYSIS DIAGNOSTIC LOG (Alignment: All Gold Annotations)")
        write(f"Generated: {datetime.now()}")
        write("=" * 70)
        
        for r in updated_results:
            sample_id = r.get("id", "UNKNOWN")
            iou_score = r.get("iou", 0.0)
            
            log.write(f"\n--- IMAGE ID: {sample_id} ---\n")
            log.write(f"MODEL PERFORMANCE (IoU): {iou_score:.4f}\n")
            
            
            # FIX: Retrieve the correct response_text for THIS specific sample
            original_row = originals.get(sample_id, {})
            current_response_text = original_row.get("response", "")
            
            log.write("GOLD LABELS (All Annotations):\n")
            for label in r.get("gold_labels_unfiltered", []):
                start, end = label.get("start", 0), label.get("end", 0)
                # Use current_response_text instead of the global/stale response_text
                span_text = current_response_text[start:end] 
                log.write(f"  [{start}:{end}] '{span_text}' | {label.get('type', 'N/A')} | agreement: {label.get('agreement', 0.0)}\n")
            # 3. ADD THIS SECTION TO LOG PREDICTED LABELS
            log.write("PREDICTED LABELS:\n")
            pred_labels = r.get("pred_labels", [])
            if pred_labels:
                for label in pred_labels:
                    start, end = label.get("start", 0), label.get("end", 0)
                    # Slice the same response text here
                    span_text = current_response_text[start:end]
                    log.write(f"  [{start}:{end}] '{span_text}' | {label.get('label', 'N/A')}\n")
            else:
                log.write("  (No spans predicted)\n")
if __name__ == "__main__":
    execute_complete_analysis()
    print("Execution: Finished.")