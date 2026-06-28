import json
from src.evaluate_row import evaluate_single_row

test_row = {
    "id": "train-en-451",
    "prompt": "Is this broccoli green in color? Please elaborate.",
    "image_name": "broccoli_purple_sprouting_broccoli_early-purple-sprouting-broccoli-grow.jpg",
    "response": "Certainly! Let's analyze the image. \n\nBased on the image, this is not the typical bright green broccoli we usually see. It's **purplish-green**. \n\nThis is likely **purple sprouting broccoli**. It's a variety of broccoli where the florets develop a beautiful purple hue, especially when exposed to sunlight. The color can fade to a greenish color when cooked. \n\nSo, while it has green tones, it's not "
}

gold_labels = [
    {"start": 136, "end": 141, "prob": 0.33, "label": "mischaracterization"},
    {"start": 378, "end": 389, "prob": 0.33, "label": "mischaracterization"},
    {"start": 448, "end": 453, "prob": 0.33, "label": "mischaracterization"}
]

print("Running broccoli test...")
result = evaluate_single_row(test_row)

print("\n--- SYSTEM OUTPUT ---")
print(json.dumps(result, indent=2))

print("\n--- GOLD LABELS ---")
for label in gold_labels:
    span_text = test_row["response"][label["start"]:label["end"]]
    print(f"  '{span_text}' | {label['label']} | prob: {label['prob']}")

print("\n--- HAIKU REASONING ---")
print(result.get("haiku_reasoning", "not saved"))