"""
Build a genuinely fresh held-out batch for the Sonnet-verification catch-rate
check: pick train-en rows never touched by scoring_run, old_prompt_v1, or
fresh_validation, run them through the CURRENT Haiku pipeline live (new API
calls), then extract mischaracterization FP/TP candidates from the results.
"""
import json
import os
import random

from src.prompt_builder import build_audit_prompt
from src.vision_llm import call_vision_llm
from src.span_mapper import map_spans_to_characters

TRAIN_PATH = r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl"
IMAGES_DIR = r"D:\SHROOM\distrib\images\shroom-vis-images"
RAW_OUT_PATH = r"D:\SHROOM\fresh_verify_raw_cache.json"

N_CLEAN = 20
N_MISCHAR = 20
SEED = 777


def get_filename_hint(image_name):
    import re
    clean = re.sub(r'[\-_.]', ' ', re.sub(r'\.(jpg|jpeg|png|bmp|gif|webp)$', '', image_name.lower())).strip()
    return f"Background context only — image filename: {clean}. Only use visual evidence." if clean else "No hint"


def main():
    all_rows = [json.loads(l) for l in open(TRAIN_PATH, encoding="utf-8")]
    excluded = set()
    for path in [r"D:\SHROOM\scoring_run\predictions.jsonl",
                 r"D:\SHROOM\old_prompt_v1_scoring_run\predictions.jsonl"]:
        for l in open(path, encoding="utf-8"):
            excluded.add(json.loads(l)["id"])
    for item in json.load(open(r"D:\SHROOM\fresh_validation_raw_cache.json", encoding="utf-8")):
        excluded.add(item["id"])

    fresh_pool = [r for r in all_rows if r["id"] not in excluded]
    clean_pool = [r for r in fresh_pool if not r.get("labels")]
    mischar_pool = [r for r in fresh_pool if any(l["label"] == "mischaracterization" for l in r.get("labels", []))]

    random.seed(SEED)
    random.shuffle(clean_pool)
    random.shuffle(mischar_pool)
    batch = clean_pool[:N_CLEAN] + mischar_pool[:N_MISCHAR]
    print(f"fresh clean rows available: {len(clean_pool)}, sampled: {min(N_CLEAN,len(clean_pool))}")
    print(f"fresh mischar rows available: {len(mischar_pool)}, sampled: {min(N_MISCHAR,len(mischar_pool))}")
    print(f"total fresh rows to call: {len(batch)}")

    cache_data = []
    if os.path.exists(RAW_OUT_PATH):
        cache_data = json.load(open(RAW_OUT_PATH, encoding="utf-8"))
    done_ids = {d["id"] for d in cache_data}

    for i, row in enumerate(batch, 1):
        if row["id"] in done_ids:
            continue
        image_path = os.path.join(IMAGES_DIR, row.get("image_name", ""))
        if not os.path.exists(image_path):
            cache_data.append({"id": row["id"], "haiku_reasoning": "[]", "gold_labels": row.get("labels", [])})
            continue
        prompt = build_audit_prompt(row.get("prompt", ""), row.get("response", ""), get_filename_hint(row.get("image_name", "")))
        try:
            _, raw = call_vision_llm(image_path, prompt)
        except Exception as e:
            print(f"  [{i}/{len(batch)}] {row['id']}: API error {e}")
            continue
        cache_data.append({"id": row["id"], "haiku_reasoning": raw, "gold_labels": row.get("labels", [])})
        print(f"  [{i}/{len(batch)}] {row['id']} done")
        if i % 15 == 0:
            json.dump(cache_data, open(RAW_OUT_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    json.dump(cache_data, open(RAW_OUT_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved {len(cache_data)} entries to {RAW_OUT_PATH}")


if __name__ == "__main__":
    main()
