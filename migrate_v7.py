import json
import os
from collections import defaultdict

# Paths
ORIGINALS_PATH = r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl"
OLD_V7_PATH = r"D:\SHROOM\v7_simulated_evaluation_results.json"
OUTPUT_DIR = r"D:\SHROOM\cache_outputs"

def migrate_v7():
    # Load your originals to lookup categories
    originals = {}
    with open(ORIGINALS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            originals[row["id"]] = row

    # Load your old v7 data
    with open(OLD_V7_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Group by category
    grouped = defaultdict(list)
    for item in data:
        row = originals.get(item["id"], {})
        labels = row.get("labels", [])
        cat = labels[0].get("label", "other").lower() if labels else "none"
        grouped[cat].append(item)
    
    # Save into new folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for cat, items in grouped.items():
        path = os.path.join(OUTPUT_DIR, f"cache_{cat}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
            
    print(f"Migration complete! Files are now in: {OUTPUT_DIR}")

if __name__ == "__main__":
    migrate_v7()