import json
import os
import subprocess

def view_sample(sample_id, 
                results_path=r"D:\SHROOM\evaluation_results.json",
                data_path=r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl",
                images_dir=r"D:\SHROOM\distrib\images\shroom-vis-images"):
    
    # Load results
    with open(results_path) as f:
        results = {r["id"]: r for r in json.load(f)}
    
    # Load original data
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
    
    # Print everything
    print(f"\n{'='*60}")
    print(f"ID: {sample_id}")
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
    
    # Open image automatically
    image_path = os.path.join(images_dir, original["image_name"])
    if os.path.exists(image_path):
        print(f"\nOpening image: {image_path}")
        os.startfile(image_path)
    else:
        print(f"\nImage not found: {image_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        view_sample(sys.argv[1])
    else:
        print("Usage: python -m src.view_sample train-en-450")