import json
import os
from src.evaluate_row import evaluate_single_row


def run_batch_evaluation(input_jsonl_path, output_jsonl_path, limit_rows=None):
    """
    Loops through a JSONL dataset line-by-line, evaluates each row, 
    and writes the final predictions instantly to protect against crashes.
    """
    if not os.path.exists(input_jsonl_path):
        print(f"❌ Error: Input file not found at {input_jsonl_path}")
        return

    print(f"📖 Loading dataset from {input_jsonl_path}...")
    
    results = []
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    
    # Apply the test limit if specified
    if limit_rows:
        results = results[:limit_rows]
        total_rows = len(results)
        print(f"⚠️ TEST MODE active: Limited to the first {total_rows} rows.")
    else:
        total_rows = len(results)
        print(f"🚀 Found {total_rows} rows. Starting evaluation...")
    
    # 🔥 Open the file BEFORE the loop starts to write on the fly
    print(f"💾 Streaming predictions directly to {output_jsonl_path}...")
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for idx, row in enumerate(results):
            prediction = evaluate_single_row(row)
            
            # Write this specific row to the hard drive immediately
            f.write(json.dumps(prediction, ensure_ascii=False) + '\n')
            f.flush()  # Force Windows/Linux to lock the data into the file right now
            
            if (idx + 1) % 10 == 0 or (idx + 1) == total_rows:
                print(f"✅ Processed {idx + 1}/{total_rows} rows...")

    print("🎉 Batch evaluation complete! Your file is safe and ready.")

if __name__ == "__main__":
    INPUT_PATH = "shroom-visions-data/distrib/shroom-vision.train.en.labeled.jsonl"
    OUTPUT_PATH = "shroom-visions-data/distrib/submission_predictions.jsonl"
    
    # 💡 SET TO 50 FOR A QUICK TEST. Change to None when you want the full run!
    TEST_LIMIT = 50 
    
    run_batch_evaluation(INPUT_PATH, OUTPUT_PATH, limit_rows=TEST_LIMIT)