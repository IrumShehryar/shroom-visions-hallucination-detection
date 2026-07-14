"""
Tests ONLY the invention-example addition, isolated from everything else
we tried before, against known "genuine miss" cases -- responses containing
confidently-stated but unverifiable claims that the ORIGINAL prompt missed.
"""

import json
import os

from src.prompt_builder import build_audit_prompt  # the isolated variant
from src.vision_llm import call_vision_llm
from src.span_mapper import map_spans_to_characters

DATA_PATH = r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl"
IMAGES_DIR = r"D:\SHROOM\distrib\images\shroom-vis-images"

# Known cases of the exact pattern this change targets: confidently-stated,
# unverifiable claims that the ORIGINAL prompt missed entirely (no prediction).
TEST_CASES = {
    "train-en-1245": "REGRESSION -- self-contradiction narration bug, should be fixed now",
    "train-en-419": "SMALL REGRESSION -- check if same narration pattern",
    "train-en-416": "SMALL REGRESSION -- check if same narration pattern",
    "train-en-11649": "SMALL REGRESSION -- check if same narration pattern",
    "train-en-2623": "KNOWN WIN -- confirm still works after this fix",
    "train-en-11340": "KNOWN WIN -- confirm still works after this fix",
}
# Add more ids here from your error_analysis_other.txt / invention.txt
# "no spans predicted" zero-IoU list for a fuller test, e.g.:
# "train-en-XXXX": "GENUINE MISS -- ...",


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

    for sample_id, note in TEST_CASES.items():
        row = originals.get(sample_id)
        if not row:
            print(f"{sample_id}: not found")
            continue

        response_text = row.get("response", "")
        gold_labels = row.get("labels", [])
        image_path = os.path.join(IMAGES_DIR, row.get("image_name", ""))

        print(f"\n{'='*70}")
        print(f"ID: {sample_id}")
        print(f"KNOWN CONTEXT: {note}")

        if not os.path.exists(image_path):
            print("Image not found, skipping")
            continue

        audit_prompt = build_audit_prompt(
            prompt_text=row.get("prompt", ""),
            response_text=response_text,
            filename_hint=get_filename_hint(row.get("image_name", "")),
        )
        llm_output, raw_response = call_vision_llm(image_path, audit_prompt)
        pred_labels = map_spans_to_characters(llm_output, response_text)
        iou = calculate_iou(pred_labels, gold_labels)

        print(f"\nISOLATED-PROMPT PREDICTED LABELS:")
        for p in pred_labels:
            print(f"  '{response_text[p['start']:p['end']]}' | {p['label']} | confidence: {p.get('prob')}")

        print(f"\nISOLATED-PROMPT IoU: {iou:.4f}  (original prompt IoU on this sample: 0.0000)")


if __name__ == "__main__":
    main()