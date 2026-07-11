import json
import os

TEST_DATA_PATH = r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.test.en.unlabeled.jsonl"
RAW_CACHE_PATH = r"D:\SHROOM\test_predictions_raw_cache.json"
IMAGES_DIR = r"D:\SHROOM\distrib\images\shroom-vis-images"


def view_test_sample(sample_id):
    original = None
    with open(TEST_DATA_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] == sample_id:
                original = row
                break

    if not original:
        print(f"Sample {sample_id} not found")
        return

    with open(RAW_CACHE_PATH, encoding="utf-8") as f:
        cache = {item["id"]: item for item in json.load(f)}

    cached_item = cache.get(sample_id, {})
    response = original["response"]

    print(f"\n{'='*60}")
    print(f"ID: {sample_id}")
    print(f"IMAGE: {original['image_name']}")
    print(f"PROMPT: {original['prompt']}")
    print(f"\nFULL RESPONSE:\n{response}")
    print(f"\nHAIKU RAW OUTPUT:\n{cached_item.get('haiku_reasoning', 'not cached')}")

    image_path = os.path.join(IMAGES_DIR, original["image_name"])
    if os.path.exists(image_path):
        print(f"\nOpening image: {image_path}")
        os.startfile(image_path)
    else:
        print(f"\nImage not found: {image_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        view_test_sample(sys.argv[1])
    else:
        print("Usage: python -m src.view_test_sample test-en-428")