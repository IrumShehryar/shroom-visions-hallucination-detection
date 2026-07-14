import json
import os
import csv
from datetime import datetime

REVIEW_LOG_PATH = r"D:\SHROOM\manual_review_log.csv"

VERDICT_OPTIONS = {
    "1": "gold_wrong",           # Haiku was right, gold label appears wrong/noisy
    "2": "haiku_hallucinated",   # Haiku's OWN flagged claim was itself false
    "3": "genuine_miss",         # Haiku missed a real error gold correctly caught
    "4": "valid_alternative",    # Haiku caught a real, different error than gold's exact span
    "5": "correct_match",        # Haiku and gold substantially agree
    "6": "ambiguous",            # genuinely unclear / low-confidence judgment call
    "skip": "skip",              # don't log anything, just move on
}


def log_verdict(sample_id, category, iou, verdict, notes=""):
    file_exists = os.path.exists(REVIEW_LOG_PATH)
    with open(REVIEW_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "id", "category", "iou", "verdict", "notes"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), sample_id, category, iou, verdict, notes])
    print(f"\nLogged: {sample_id} -> {verdict}")


def view_sample(sample_id,
                results_path=r"D:\SHROOM\v1_analysis_outputs\all_results.json",
                data_path=r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl",
                images_dir=r"D:\SHROOM\distrib\images\shroom-vis-images",
                interactive=True):

    with open(results_path, encoding="utf-8") as f:
        results = {r["id"]: r for r in json.load(f)}

    original = None
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] == sample_id:
                original = row
                break

    if not original:
        print(f"Sample {sample_id} not found")
        return

    result = results.get(sample_id, {})
    response = original["response"]

    gold_labels_orig = original.get("labels", [])
    category = gold_labels_orig[0].get("label", "other").lower() if gold_labels_orig else "none"

    print(f"\n{'='*60}")
    print(f"ID: {sample_id}")
    print(f"CATEGORY: {category}")
    print(f"IMAGE: {original['image_name']}")
    print(f"PROMPT: {original['prompt']}")
    print(f"IoU: {result.get('iou', 'N/A')}")
    print(f"\nFULL RESPONSE:\n{response}")

    print(f"\nGOLD LABELS:")
    for label in result.get("gold_labels", []):
        span = response[label["start"]:label["end"]]
        print(f"  '{span}' | {label['label']} | agreement: {label['prob']}")

    print(f"\nPREDICTED LABELS:")
    for label in result.get("pred_labels", []):
        span = response[label["start"]:label["end"]]
        print(f"  '{span}' | {label['label']} | confidence: {label['prob']}")

    print(f"\nHAIKU REASONING:\n{result.get('haiku_reasoning', 'not saved')}")

    image_path = os.path.join(images_dir, original["image_name"])
    if os.path.exists(image_path):
        print(f"\nOpening image: {image_path}")
        os.startfile(image_path)
    else:
        print(f"\nImage not found: {image_path}")

    if interactive:
        print("\n" + "=" * 60)
        print("YOUR VERDICT (after looking at the image):")
        for key, val in VERDICT_OPTIONS.items():
            if key != "skip":
                print(f"  {key} = {val}")
        print("  skip = don't log, just move on")
        choice = input("Enter verdict: ").strip().lower()

        verdict = VERDICT_OPTIONS.get(choice)
        if verdict and verdict != "skip":
            notes = input("Optional note (press Enter to skip): ").strip()
            log_verdict(sample_id, category, result.get("iou", "N/A"), verdict, notes)
        else:
            print("Skipped, nothing logged.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        view_sample(sys.argv[1])
    else:
        print("Usage: python -m src.view_sample train-en-450")