import json
import os

def calculate_confusion_matrix(ground_truth_path, predictions_path):
    """
    Compares the generated char_probabilities against human ground truth 
    at the character level to calculate Precision, Recall, and F1-Score.
    """
    if not os.path.exists(ground_truth_path) or not os.path.exists(predictions_path):
        print("❌ Error: One or both files are missing!")
        print(f"Checking Ground Truth: {os.path.exists(ground_truth_path)} ({ground_truth_path})")
        print(f"Checking Predictions: {os.path.exists(predictions_path)} ({predictions_path})")
        return

    # Load Ground Truth mapped by ID
    gt_data = {}
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                gt_data[row["id"]] = row

    # Load Predictions mapped by ID
    pred_data = {}
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                pred_data[row["id"]] = row

    # Initialize counters for character-level Confusion Matrix
    true_positives = 0   # Code flagged hallucination, Human said YES
    false_positives = 0  # Code flagged hallucination, Human said NO
    false_negatives = 0  # Code missed it, Human said YES
    true_negatives = 0   # Code said clean, Human said NO (clean)

    matched_rows = 0

    for row_id, gt_row in gt_data.items():
        if row_id not in pred_data:
            continue
        
        matched_rows += 1
        pred_row = pred_data[row_id]
        response_text = gt_row.get("response", "")
        response_len = len(response_text)
        
        # 1. Reconstruct Ground Truth binary array from its span offsets
        gt_array = [0] * response_len
        for span in gt_row.get("labels", []):
            start = span["start"]
            end = span["end"]
            prob = span["prob"]
            # If at least one human flagged it (prob > 0), treat it as a true hallucination character
            if prob > 0.0:
                for i in range(start, min(end, response_len)):
                    gt_array[i] = 1
        
        # 2. Get our model predictions array
        pred_probs = pred_row.get("char_probabilities", [0.0] * response_len)
        
        # Ensure lengths match up safely to protect against unexpected mismatches
        min_len = min(len(gt_array), len(pred_probs))

        # 3. Step through and calculate character alignment
        for i in range(min_len):
            gt_char = gt_array[i]
            pred_char = 1 if pred_probs[i] > 0.0 else 0

            if pred_char == 1 and gt_char == 1:
                true_positives += 1
            elif pred_char == 1 and gt_char == 0:
                false_positives += 1
            elif pred_char == 0 and gt_char == 1:
                false_negatives += 1
            elif pred_char == 0 and gt_char == 0:
                true_negatives += 1

    print(f"\n📊 --- EVALUATION REPORT ({matched_rows} Rows Processed) ---")
    print(f"True Positives (TP):  {true_positives}")
    print(f"False Positives (FP): {false_positives}")
    print(f"False Negatives (FN): {false_negatives}")
    print(f"True Negatives (TN):  {true_negatives}\n")

    # Calculate standard macro metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"🎯 Precision: {precision:.4f} (When code flags an error, how often it's right)")
    print(f"📢 Recall:    {recall:.4f} (How many of the actual errors your code caught)")
    print(f"🏆 F1-Score:  {f1_score:.4f} (Your overall baseline competition grade)")

if __name__ == "__main__":
    GROUND_TRUTH = "shroom-visions-data/distrib/shroom-vision.train.en.labeled.jsonl"
    PREDICTIONS = "shroom-visions-data/distrib/submission_predictions.jsonl"
    
    calculate_confusion_matrix(GROUND_TRUTH, PREDICTIONS)