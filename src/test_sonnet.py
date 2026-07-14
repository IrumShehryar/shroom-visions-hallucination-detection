"""
Tiny, cheap diagnostic: runs a handful of KNOWN cases (already manually
verified against the real image) through Sonnet instead of Haiku, to see
if a stronger model fixes the specific failure types you've documented.

Costs ~4 API calls total (Sonnet pricing: $2/$10 per million tokens,
introductory through Aug 31 2026 -- roughly $0.01-0.02 for this whole test).
"""

import json
import os
import base64
import anthropic
from dotenv import load_dotenv

from src.prompt_builder import build_audit_prompt
from src.span_mapper import map_spans_to_characters
from src.vision_llm import parse_llm_response, load_image_as_base64, get_image_media_type

load_dotenv()

DATA_PATH = r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl"
IMAGES_DIR = r"D:\SHROOM\distrib\images\shroom-vis-images"

# id -> what you already know about it, for reference while reading results
TEST_CASES = {
    "train-en-9920": "FALSE POSITIVE -- gold empty, Haiku wrongly claimed only 1 bench not 3",
    "train-en-460": "FALSE POSITIVE -- Haiku wrongly denied a cross that IS in the image",
    "train-en-2533": "GENUINE MISS -- Haiku missed fabricated grape-seed biology claims",
    "train-en-1029": "PARTIAL MATCH -- screws/adhesive mischaracterization, boundary mismatch",
}


def get_filename_hint(image_name):
    import re
    clean = re.sub(r'[\-_.]', ' ', re.sub(r'\.(jpg|jpeg|png|bmp|gif|webp)$', '', image_name.lower())).strip()
    return f"Background context only — image filename: {clean}. Only use visual evidence." if clean else "No hint"


def call_sonnet(image_path, audit_prompt):
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    image_data = load_image_as_base64(image_path)
    media_type = get_image_media_type(image_path)

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                {"type": "text", "text": audit_prompt},
            ],
        }],
    )
    # Sonnet may return a ThinkingBlock before the TextBlock -- find the
    # actual text block instead of assuming it's content[0].
    raw_text = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            raw_text = block.text
            break

    print(f"USAGE: input_tokens={response.usage.input_tokens}, output_tokens={response.usage.output_tokens}")
    return parse_llm_response(raw_text), raw_text


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

        sonnet_output, sonnet_raw = call_sonnet(image_path, audit_prompt)
        sonnet_pred = map_spans_to_characters(sonnet_output, response_text)
        sonnet_iou = calculate_iou(sonnet_pred, gold_labels)

        print(f"\nSONNET PREDICTED LABELS:")
        for p in sonnet_pred:
            print(f"  '{response_text[p['start']:p['end']]}' | {p['label']} | confidence: {p.get('prob')}")

        print(f"\nSONNET IoU: {sonnet_iou:.4f}")


if __name__ == "__main__":
    main()