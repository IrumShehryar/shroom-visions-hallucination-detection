"""
Pulls a target number of UNPROCESSED samples per hallucination category,
calls the Haiku API, and saves the RAW results into separate category-specific files.
"""

import json
import os
import random
from collections import defaultdict

from src.prompt_builder import build_audit_prompt
from src.vision_llm import call_vision_llm

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
FULL_DATA_PATH = r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl"
OUTPUT_DIR = r"D:\SHROOM\cache_outputs"  # New directory for split files
IMAGES_DIR = r"D:\SHROOM\distrib\images\shroom-vis-images"

TARGET_PER_CATEGORY = 50
CATEGORIES = ["invention", "mischaracterization", "miscounting", "ocr", "other", "none"]
RANDOM_SEED = 42
# ---------------------------------------------------------------------------

def get_category_path(category):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, f"cache_{category}.json")

def load_category_cache(category):
    path = get_category_path(category)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_category_cache(category, cache_data):
    path = get_category_path(category)
    # Simple backup
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as src, open(path + ".bak", "w", encoding="utf-8") as dst:
            dst.write(src.read())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)

def primary_category(row):
    labels = row.get("labels", [])
    if not labels: return "none"
    return labels[0].get("label", "other").lower()

def load_full_dataset(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def get_filename_hint(image_name):
    # (Keeping your original helper logic)
    import re
    clean = re.sub(r'[\-_.]', ' ', re.sub(r'\.(jpg|jpeg|png|bmp|gif|webp)$', '', image_name.lower())).strip()
    return f"Background context only — image filename: {clean}. Only use visual evidence." if clean else "No hint"

def run_batch(all_rows, target_per_category, images_dir):
    # 1. Group all unprocessed rows by category
    by_category = defaultdict(list)
    for row in all_rows:
        cat = primary_category(row)
        # Check if already processed in the specific category file
        already_done = {item["id"] for item in load_category_cache(cat)}
        if row["id"] not in already_done:
            by_category[cat].append(row)

    # 2. Select samples and process
    for cat in CATEGORIES:
        pool = by_category.get(cat, [])
        random.seed(RANDOM_SEED)
        random.shuffle(pool)
        batch = pool[:target_per_category]
        
        if not batch:
            print(f"No new samples to process for {cat}.")
            continue

        print(f"\n--- Processing Category: {cat} ({len(batch)} samples) ---")
        cache_data = load_category_cache(cat)

        for i, row in enumerate(batch, 1):
            print(f"[{i}/{len(batch)}] Calling API for {row['id']}...")
            image_path = os.path.join(images_dir, row.get("image_name", ""))
            
            audit_prompt = build_audit_prompt(row.get("prompt", ""), row.get("response", ""), get_filename_hint(row.get("image_name", "")))
            
            try:
                _, raw_response = call_vision_llm(image_path, audit_prompt)
                cache_data.append({
                    "id": row["id"],
                    "gold_labels": row.get("labels", []),
                    "haiku_reasoning": raw_response,
                })
            except Exception as e:
                print(f"  Error on {row['id']}: {e}")
                continue

            if i % 10 == 0:
                save_category_cache(cat, cache_data)
        
        save_category_cache(cat, cache_data)
        print(f"Saved {len(cache_data)} total samples for {cat}.")

def main():
    all_rows = load_full_dataset(FULL_DATA_PATH)
    print(f"Full dataset: {len(all_rows)} samples loaded.")
    
    confirm = input("Proceed with API calls to category-specific files? (y/n): ").strip().lower()
    if confirm == "y":
        run_batch(all_rows, TARGET_PER_CATEGORY, IMAGES_DIR)
    else:
        print("Aborted.")

if __name__ == "__main__":
    main()