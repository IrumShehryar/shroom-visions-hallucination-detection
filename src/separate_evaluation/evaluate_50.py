import json
from src.evaluate_row import evaluate_single_row
def calculate_iou(pred_spans, gold_spans, response_len, agreement_threshold=0.5):
    pred_chars = set()
    gold_chars = set()
    
    for span in pred_spans:
        for i in range(span["start"], span["end"]):
            pred_chars.add(i)
    
    # Filters out human noise under the threshold
    for span in gold_spans:
        prob = span.get("prob", 1.0)
        if prob >= agreement_threshold:
            for i in range(span["start"], span["end"]):
                gold_chars.add(i)
    
    if not pred_chars and not gold_chars:
        return 1.0
    
    if not pred_chars or not gold_chars:
        return 0.0
    
    intersection = len(pred_chars & gold_chars)
    union = len(pred_chars | gold_chars)
    
    return intersection / union


def run_evaluation(input_path, limit=50):
    results = []
    iou_scores = []
    originals = {}
    
    # -------------------------------------------------------------
    # 🛠️ QUICK CONTROLLER: Add your target IDs here to isolate them.
    # Leave it empty like this: TARGET_IDS = [] to run everything normally!
    # -------------------------------------------------------------
    TARGET_IDS = []  # Example: ["train-en-415", "train-en-416"]
    
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            originals[row["id"]] = row
    
    category_correct = {"miscounting": 0, "mischaracterization": 0, 
                        "invention": 0, "ocr": 0, "other": 0}
    category_total = {"miscounting": 0, "mischaracterization": 0, 
                      "invention": 0, "ocr": 0, "other": 0}

    evaluated_count = 0
    with open(input_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            row = json.loads(line.strip())
            
            # If targeting specific IDs, skip everything else
            if TARGET_IDS and row["id"] not in TARGET_IDS:
                continue
                
            # If running normally (no specific targets), honor the standard limit
            if not TARGET_IDS and evaluated_count >= limit:
                break
            
            gold_labels = row.get("labels", [])
            print(f"Processing ({evaluated_count + 1}): {row['id']}...")
            
            prediction = evaluate_single_row(row)
            pred_labels = prediction.get("labels", [])
            
            # Calculate IoU
            response_len = len(row["response"])
            iou = calculate_iou(pred_labels, gold_labels, response_len, agreement_threshold=0.5)
            iou_scores.append(iou)
            
            # Track by category
            for label in gold_labels:
                cat = label["label"].lower().replace(" ", "_")
                if cat in category_total:
                    category_total[cat] += 1
                    pred_cats = [p["label"].lower() for p in pred_labels]
                    if cat in pred_cats:
                        category_correct[cat] += 1
            
            results.append({
                "id": row["id"],
                "iou": round(iou, 4),
                "pred_count": len(pred_labels),
                "gold_count": len(gold_labels),
                "pred_labels": pred_labels,
                "gold_labels": gold_labels,
                "haiku_reasoning": prediction.get("haiku_reasoning", "")
            })
            
            print(f"  IoU: {iou:.4f} | Pred spans: {len(pred_labels)} | Gold spans: {len(gold_labels)}")
            evaluated_count += 1

    # Summary calculations (safeguarded against zero division if running few samples)
    total_evaluated = len(iou_scores) if iou_scores else 1
    avg_iou = sum(iou_scores) / total_evaluated
    perfect = sum(1 for s in iou_scores if s == 1.0)
    zero = sum(1 for s in iou_scores if s == 0.0)
    
    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY ({len(iou_scores)} samples processed)")
    print(f"{'='*50}")
    print(f"Average IoU:     {avg_iou:.4f}")
    print(f"Perfect (1.0):   {perfect} samples")
    print(f"Zero (0.0):      {zero} samples")
    print(f"\nBy category (detection rate):")
    for cat in category_total:
        total = category_total[cat]
        correct = category_correct[cat]
        if total > 0:
            print(f"  {cat}: {correct}/{total} ({correct/total*100:.1f}%)")
    
    output_path = r"D:\SHROOM\evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")
    
    csv_path = r"D:\SHROOM\evaluation_results.csv"
    save_csv(results, originals, csv_path)
    print(f"CSV saved to: {csv_path}")
    
    return results

def save_csv(results, originals, output_path):
    import csv
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "iou", "pred_count", "gold_count",
            "gold_categories", "pred_categories",
            "gold_probs", "pred_probs",
            "prompt", "haiku_reasoning"
        ])
        for r in results:
            original = originals.get(r["id"], {})
            gold_cats = "|".join([l["label"] for l in r["gold_labels"]])
            pred_cats = "|".join([l["label"] for l in r["pred_labels"]])
            gold_probs = "|".join([str(round(l["prob"], 2)) for l in r["gold_labels"]])
            pred_probs = "|".join([str(round(l["prob"], 2)) for l in r["pred_labels"]])
            writer.writerow([
                r["id"],
                r["iou"],
                r["pred_count"],
                r["gold_count"],
                gold_cats,
                pred_cats,
                gold_probs,
                pred_probs,
                original.get("prompt", ""),
                r.get("haiku_reasoning", "")
            ])
if __name__ == "__main__":
    INPUT = r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl"
    results = run_evaluation(INPUT, limit=50)