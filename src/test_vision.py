import json
from src.evaluate_row import evaluate_single_row

test_cases = [
    {
    "row": {
        "id": "train-en-416",
        "prompt": "Does this cat have four legs? Please elaborate.",
        "image_name": "cat_Three-legged_cat_3-legged-cat-sitting_Ted-Arts-Shutte.jpg",
        "response": "Yes, this cat appears to have four legs. \n\nIn the image, we can clearly see all four of the cat's legs: two front legs supporting its upper body and two back legs positioned beneath its rear. While part of the back legs are hidden by the cat's body, their presence is evident. \n\nIt's a healthy-looking cat!"
    },
    "gold": [
        {"start": 30, "end": 31, "prob": 0.67, "label": "miscounting"},
        {"start": 31, "end": 34, "prob": 1.0, "label": "miscounting"},
        {"start": 80, "end": 84, "prob": 0.67, "label": "miscounting"},
        {"start": 104, "end": 107, "prob": 0.67, "label": "miscounting"}
    ]
},

   """ {
        "row": {
            "id": "train-en-413",
            "prompt": "Does this desk have legs? Please elaborate.",
            "image_name": "desk_Floating_desk_ab0a79a9b6c3e5e2578ef6d938ad5e8f.jpg",
            "response": "No, this desk does not have legs in the traditional sense. \n\nIt's a **floating desk**, meaning it's mounted directly to the wall and supported by a hidden bracket system and cables. You can see the cables attached to the ceiling providing additional support. \n\nInstead of legs resting on the floor, it relies on the wall's structure to hold it up. This gives it a minimalist and modern look, and it's a good space-saving option!"
        },
        "gold": [
            {"start": 148, "end": 154, "prob": 0.33, "label": "mischaracterization"},
            {"start": 221, "end": 228, "prob": 0.33, "label": "mischaracterization"}
        ]
    },
    {
        "row": {
            "id": "train-en-411",
            "prompt": "Does this ship have masts? Please elaborate.",
            "image_name": "ship_Yachts_and_luxury_vessel_1-Profile-print-2.jpg",
            "response": "Based on the image, the ship does **not** have traditional masts like those found on sailing vessels. \n\nHowever, it does have a **communications mast** \u2013 a relatively small, single pole structure visible towards the top of the ship. This mast is used for antennas and communication equipment, not for sails. It's a common feature on modern motor yachts like this one.\n\nSo, while it doesn't have masts in the traditional sense, it does have a mast-like structure for technical purposes."
        },
        "gold": []
    },
    
    {
        "row": {
            "id": "train-en-435",
            "prompt": "What is the sea name?",
            "image_name": "7691092790_3e2186a48f_o.jpg",
            "response": "Based on the image, this appears to be the Devonport ferry terminal in Auckland, New Zealand. Therefore, the sea is the **Hauraki Gulf**. \n\nIt's a large gulf near Auckland, and the ferry in the image is likely a ferry that travels across it."
        },
        "gold": [
            {"start": 0, "end": 43, "prob": 0.33, "label": "invention"},
            {"start": 43, "end": 52, "prob": 0.67, "label": "invention"},
            {"start": 52, "end": 71, "prob": 0.33, "label": "invention"},
            {"start": 71, "end": 92, "prob": 0.67, "label": "invention"},
            {"start": 92, "end": 122, "prob": 0.33, "label": "invention"},
            {"start": 122, "end": 134, "prob": 0.67, "label": "invention"},
            {"start": 134, "end": 163, "prob": 0.33, "label": "invention"},
            {"start": 163, "end": 171, "prob": 0.67, "label": "invention"},
            {"start": 171, "end": 240, "prob": 0.33, "label": "invention"}
        ]
    },
    {
        "row": {
            "id": "train-en-426",
            "prompt": "What is written on the projector screen?",
            "image_name": "2303189378_87dc77f519_o.jpg",
            "response": "Here's what is written on the projector screen, as best as I can read it from the image:\n\n**What is OpenID?**\n\n*   A system that enables users to authenticate once to log in everywhere.\n*   Instead of creating an account on site X, you log in with your OpenID URL. You sign on to your OpenID provider and verify that you want site X to be able to access your credentials.\n\n"
        },
        "gold": [
            {"start": 236, "end": 239, "prob": 0.33, "label": "OCR"},
            {"start": 269, "end": 276, "prob": 0.67, "label": "OCR"},
            {"start": 305, "end": 311, "prob": 0.67, "label": "OCR"}
        ]
    }"""
]


def print_comparison(result, gold, response):
    print(f"\n{'='*50}")
    print(f"ID: {result['id']}")
    print(f"\nSYSTEM OUTPUT:")
    if result["labels"]:
        for label in result["labels"]:
            span_text = response[label["start"]:label["end"]]
            print(f"  '{span_text}' | {label['label']} | prob: {label['prob']}")
    else:
        print("  No hallucinations detected")

    print(f"\nGOLD LABELS:")
    if gold:
        for label in gold:
            span_text = response[label["start"]:label["end"]]
            print(f"  '{span_text}' | {label['label']} | prob: {label['prob']}")
    else:
        print("  No hallucinations (empty)")

    correct = "✓" if (not result["labels"] and not gold) else "~"
    print(f"\nMATCH: {correct}")


for case in test_cases:
    print(f"\nRunning {case['row']['id']}...")
    result = evaluate_single_row(case["row"])
    print_comparison(result, case["gold"], case["row"]["response"])