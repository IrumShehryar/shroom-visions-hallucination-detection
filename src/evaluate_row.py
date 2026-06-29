import re
import os
from src.parse_text import extract_linguistic_features
from src.prompt_builder import build_audit_prompt
from src.vision_llm import call_vision_llm
from src.span_mapper import map_spans_to_characters

NUMERIC_MAPPING = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
}

def clean_and_check_filename(filename):
    filename_lower = filename.lower()
    filename_clean = re.sub(r'\.(jpg|jpeg|png|bmp|gif|webp)$', '', filename_lower)
    filename_clean = re.sub(r'[\-_.]', ' ', filename_clean).strip()
    words = filename_clean.split()
    if not words:
        return filename_clean, False
    alpha_words = [w for w in words if w.isalpha() and len(w) > 1]
    is_descriptive = (len(alpha_words) / len(words)) >= 0.2
    return filename_clean, is_descriptive

def get_filename_hint(image_name):
    clean_filename, is_descriptive = clean_and_check_filename(image_name)
    if is_descriptive:
        return f"Background context only — image filename: {clean_filename}. Do not use this to validate response claims. Only use visual evidence from the image itself."
    return "No descriptive filename available"

def get_image_path(image_name, images_dir=r"D:\SHROOM\distrib\images\shroom-vis-images"):
    return os.path.join(images_dir, image_name)

def evaluate_single_row(dataset_row):
    prompt_text = dataset_row.get("prompt", "")
    image_name = dataset_row.get("image_name", "")
    model_response = dataset_row.get("response", "")

    # Step 1: Get filename hint
    filename_hint = get_filename_hint(image_name)

    # Step 2: Build audit prompt
    audit_prompt = build_audit_prompt(
        prompt_text=prompt_text,
        response_text=model_response,
        filename_hint=filename_hint
    )

    # Step 3: Get image path
    image_path = get_image_path(image_name)

    # Step 4: Call vision LLM
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return {"id": dataset_row.get("id"), "labels": []}

    llm_output,raw_response = call_vision_llm(image_path, audit_prompt)

    # Step 5: Map spans to character offsets
    labels = map_spans_to_characters(llm_output, model_response)

    return {
        "id": dataset_row.get("id"),
        "labels": labels,
        "haiku_reasoning": raw_response
    }


