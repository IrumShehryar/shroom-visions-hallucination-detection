"""
Quick smoke test: runs a handful of KNOWN train samples (with gold labels)
through your final pipeline (reverted prompt + fixed parse_llm_response),
so you can visually confirm parsing is working correctly before spending
API budget on the blind 1200-sample test set.
"""

import json
import os

from src.prompt_builder import build_audit_prompt
from src.vision_llm import call_vision_llm
from src.span_mapper import map_spans_to_characters

DATA_PATH = r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl"
IMAGES_DIR = r"D:\SHROOM\distrib\images\shroom-vis-images"

# 3 known ids -- pick ones you've already reviewed, e.g. from your manual log
SMOKE_TEST_IDS = [
    "train-en-9920",
    "train-en-10126",   # this one previously had trailing-prose parsing issues
    "train-en-460",
]


def get_filename_hint(image_name):
    import re
    clean = re.sub(r'[\-_.]', ' ', re.sub(r'\.(jpg|jpeg|png|bmp|gif|webp)$', '', image_name.lower())).strip()
    return f"Background context only — image filename: {clean}. Only use visual evidence." if clean else "No hint"


def calculate_iou(pred_spans, gold_spans):
    pred_chars = {i for s in pred_spans for i in range(s["start"], s["end"])}
    gold_chars = {i for s in gold_spans for i in range(s["start"], s["end"])}
    if not pred_chars and not gold_chars:
        return 1.0
    if not pred_chars or not gold_chars:
        return 0.0
    return len(pred_chars & gold_chars) / len(pred_chars | gold_chars)


def main():
    originals = {}
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            originals[row["id"]] = row

    for sample_id in SMOKE_TEST_IDS:
        row = originals.get(sample_id)
        if not row:
            print(f"{sample_id}: not found, skipping")
            continue

        response_text = row.get("response", "")
        gold_labels = row.get("labels", [])
        image_path = os.path.join(IMAGES_DIR, row.get("image_name", ""))

        print(f"\n{'='*60}")
        print(f"ID: {sample_id}")

        if not os.path.exists(image_path):
            print("Image not found, skipping")
            continue

        audit_prompt = build_audit_prompt(
            prompt_text=row.get("prompt", ""),
            response_text=response_text,
            filename_hint=get_filename_hint(row.get("image_name", "")),
        )
        llm_output, raw_response = call_vision_llm(image_path, audit_prompt)

        print(f"\nPARSED llm_output (this is what matters -- should be a real list, not [] if raw text had content):")
        print(llm_output)

        pred_labels = map_spans_to_characters(llm_output, response_text)
        iou = calculate_iou(pred_labels, gold_labels)

        print(f"\nGOLD LABELS:")
        for g in gold_labels:
            print(f"  '{response_text[g['start']:g['end']]}' | {g['label']} | agreement: {g.get('prob')}")

        print(f"\nPREDICTED LABELS (after span_mapper):")
        for p in pred_labels:
            print(f"  '{response_text[p['start']:p['end']]}' | {p['label']} | confidence: {p.get('prob')}")

        print(f"\nIoU: {iou:.4f}")


if __name__ == "__main__":
    main()