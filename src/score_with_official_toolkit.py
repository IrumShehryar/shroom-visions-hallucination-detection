"""
Runs your predictions through SHROOM-Visions' OWN format_checker.py and
scorer.py. This is the number that actually matters for your paper —
not the custom calculate_iou.

Put these 4 files in the SAME folder as this script (they already are,
if you're running this where you uploaded everything):
  - format_checker.py                        (their file)
  - scorer.py                                 (their file)
  - v7_simulated_evaluation_results.json      (your Haiku cache)
  - shroom-vision.train.en.labeled.jsonl      (the real gold data)

Then just run:  python score_with_official_toolkit.py
"""

import json
from pathlib import Path

import format_checker          # their file, same folder
import scorer as official_scorer  # their file, same folder

from test_offline import map_spans_to_characters  # your span logic

# ---------------------------------------------------------------------------
# Files — all in the same folder as this script
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
CACHE_PATH = HERE / "v7_simulated_evaluation_results.json"
ORIGINALS_PATH = HERE / "shroom-vision.train.en.labeled.jsonl"
OUT_DIR = HERE / "scoring_run"


def build_official_records(cache_data, originals):
    """Build (ref_dicts, pred_dicts) in the exact schema scorer.py expects,
    sorted by id so main()'s positional zip lines them up correctly."""

    ref_dicts = []
    pred_dicts = []

    for cached_item in cache_data:
        sample_id = cached_item["id"]
        original_row = originals.get(sample_id, {})
        response_text = original_row.get("response", "")

        raw_haiku = cached_item.get("haiku_reasoning", "[]")
        raw_haiku = raw_haiku.replace("```json", "").replace("```", "").strip()
        try:
            llm_output = json.loads(raw_haiku)
        except Exception:
            llm_output = []

        pred_labels = map_spans_to_characters(llm_output, response_text)
        gold_labels = cached_item.get("gold_labels", [])

        # scorer.py's __check_one requires start/end as int, prob as float,
        # label as str from a fixed set. Also NOTE: labels are case-sensitive
        # ("OCR" not "ocr", lowercase for the rest) -- fix that here too.
        LABEL_FIX = {
            "invention": "invention", "mischaracterization": "mischaracterization",
            "ocr": "OCR", "miscounting": "miscounting", "other": "other",
        }

        def clean_span(s):
            return {
                "start": int(s["start"]),
                "end": int(s["end"]),
                "prob": float(s.get("prob", 1.0)),
                "label": LABEL_FIX.get(str(s["label"]).lower(), str(s["label"])),
            }

        clean_gold = [clean_span(g) for g in gold_labels]
        clean_pred = [clean_span(p) for p in pred_labels]

        ref_dicts.append({
            "id": sample_id,
            "labels": clean_gold,
            "text_len": len(response_text),
            # scores_classif needs raw per-annotator marks, which aren't in
            # this file -> leave empty. cor/cor_lbl/IoU stay valid either way.
            "raw_annots": original_row.get("raw_annots", {}),
        })
        pred_dicts.append({
            "id": sample_id,
            "labels": clean_pred,
        })

    # CRITICAL: sort both by id -- scorer.main() zips by position, not by id.
    ref_dicts.sort(key=lambda r: r["id"])
    pred_dicts.sort(key=lambda p: p["id"])
    return ref_dicts, pred_dicts


def write_submission_jsonl(pred_dicts, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for row in pred_dicts:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache_data = json.load(f)

    originals = {}
    with open(ORIGINALS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            originals[row["id"]] = row

    ref_dicts, pred_dicts = build_official_records(cache_data, originals)

    submission_path = OUT_DIR / "predictions.jsonl"
    write_submission_jsonl(pred_dicts, submission_path)

    print("\n=== RUNNING format_checker.py ===")
    errors, warnings, stats = format_checker.check_submission_files(
        submission_files=[submission_path],
        reference_files=[],
        reference_dirs=[],
    )
    print(f"Checked {stats['files']} file(s), {stats['rows']} row(s), {stats['spans']} span(s).")
    for w in warnings:
        print("WARNING:", w)
    for e in errors[:20]:
        print("ERROR:", e)
    print("format_checker: OK, no errors." if not errors else f"format_checker found {len(errors)} error(s).")

    print("\n=== RUNNING scorer.py ===")
    cors, cors_lbl = official_scorer.main(ref_dicts, pred_dicts, output_file=None, verbose=True)

    print(f"\nDone. {len(cors)} samples scored with the OFFICIAL scorer.py.")


if __name__ == "__main__":
    main()