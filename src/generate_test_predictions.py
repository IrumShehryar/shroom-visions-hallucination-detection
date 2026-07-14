"""
Generates final submission predictions for the SHROOM-Visions TEST set
(no gold labels available -- this is the real, final deliverable).

Uses your FINAL locked pipeline: reverted prompt_builder.py + span_mapper.py.
Calls the API exactly once per test sample, caches raw responses (so a
crash doesn't cost you re-calls), then writes the submission-format JSONL
and validates it with format_checker.py before you're done.
"""

import json
import os
from pathlib import Path

from src.prompt_builder import build_audit_prompt
from src.vision_llm import call_vision_llm
from src.span_mapper import map_spans_to_characters
from src import format_checker

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------
TEST_DATA_PATH = r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.test.en.unlabeled.jsonl"
IMAGES_DIR = r"D:\SHROOM\distrib\images\shroom-vis-images"
RAW_CACHE_PATH = r"D:\SHROOM\test_predictions_raw_cache.json"   # checkpoint, safe to resume
SUBMISSION_OUT = r"D:\SHROOM\submission\en.jsonl"
LANGUAGE = "en"
TEST_LIMIT = None   # set to None to run all 1201 samples
# ---------------------------------------------------------------------------


def get_filename_hint(image_name):
    import re
    clean = re.sub(r'[\-_.]', ' ', re.sub(r'\.(jpg|jpeg|png|bmp|gif|webp)$', '', image_name.lower())).strip()
    return f"Background context only — image filename: {clean}. Only use visual evidence." if clean else "No hint"


def load_test_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if TEST_LIMIT:
        rows = rows[:TEST_LIMIT]
    return rows


def load_raw_cache():
    if os.path.exists(RAW_CACHE_PATH):
        with open(RAW_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_raw_cache(cache_data):
    with open(RAW_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)


def run_predictions():
    rows = load_test_rows(TEST_DATA_PATH)
    print(f"Test set: {len(rows)} samples")

    cache_data = load_raw_cache()
    done_ids = {item["id"] for item in cache_data}
    print(f"Already cached (resuming, no re-call): {len(done_ids)} samples")

    remaining = [r for r in rows if r["id"] not in done_ids]
    print(f"Remaining to call: {len(remaining)} samples")

    if remaining:
        confirm = input(f"Proceed with {len(remaining)} API calls? (y/n): ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return None

    for i, row in enumerate(remaining, 1):
        sample_id = row["id"]
        image_path = os.path.join(IMAGES_DIR, row.get("image_name", ""))

        if not os.path.exists(image_path):
            print(f"[{i}/{len(remaining)}] {sample_id}: image not found, saving empty prediction")
            cache_data.append({"id": sample_id, "haiku_reasoning": "[]"})
            continue

        print(f"[{i}/{len(remaining)}] Calling API for {sample_id}...")
        audit_prompt = build_audit_prompt(
            prompt_text=row.get("prompt", ""),
            response_text=row.get("response", ""),
            filename_hint=get_filename_hint(row.get("image_name", "")),
        )
        try:
            _, raw_response = call_vision_llm(image_path, audit_prompt)
        except Exception as e:
            print(f"  API call failed: {e}. Saving empty prediction.")
            cache_data.append({"id": sample_id, "haiku_reasoning": "[]"})
            continue

        cache_data.append({"id": sample_id, "haiku_reasoning": raw_response})
        done_ids.add(sample_id)

        if i % 25 == 0:
            save_raw_cache(cache_data)
            print(f"  ...checkpoint saved ({len(cache_data)} total cached)")

    save_raw_cache(cache_data)
    print(f"\nAll API calls done. {len(cache_data)} total cached.")
    return rows


def build_submission(rows, cache_data):
    response_by_id = {r["id"]: r.get("response", "") for r in rows}
    cache_by_id = {item["id"]: item for item in cache_data}

    submission_rows = []
    for row in rows:
        sample_id = row["id"]
        response_text = response_by_id[sample_id]
        raw = cache_by_id.get(sample_id, {}).get("haiku_reasoning", "[]")
        raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            llm_output = json.loads(raw)
        except json.JSONDecodeError:
            llm_output = []

        pred_labels = map_spans_to_characters(llm_output, response_text)
        clean_labels = [
            {
                "start": int(p["start"]),
                "end": int(p["end"]),
                "prob": float(p.get("prob", 1.0)),
                "label": str(p["label"]),
            }
            for p in pred_labels
        ]
        submission_rows.append({"id": sample_id, "labels": clean_labels})

    return submission_rows


def write_and_validate(submission_rows):
    out_path = Path(SUBMISSION_OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in submission_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSubmission written to: {out_path}")

    print("\n=== RUNNING format_checker.py ===")
    errors, warnings, stats = format_checker.check_submission_files(
        submission_files=[out_path],
        reference_files=[TEST_DATA_PATH],
        reference_dirs=[],
    )
    print(f"Checked {stats['files']} file(s), {stats['rows']} row(s), {stats['spans']} span(s).")
    for w in warnings:
        print("WARNING:", w)
    for e in errors[:20]:
        print("ERROR:", e)
    if errors:
        print(f"\n{len(errors)} error(s) found -- FIX BEFORE SUBMITTING.")
    else:
        print("\nformat_checker: OK. Ready to submit.")


def main():
    rows = run_predictions()
    if rows is None:
        return
    cache_data = load_raw_cache()
    submission_rows = build_submission(rows, cache_data)
    write_and_validate(submission_rows)


if __name__ == "__main__":
    main()