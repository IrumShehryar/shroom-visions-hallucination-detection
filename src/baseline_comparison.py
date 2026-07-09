"""
Runs the SAME baseline logic as constant_baselines.py (mark_none, and
mark_<label> for each category), but evaluated on the exact same samples
you already scored with your own model -- so it's a fair, apples-to-apples
comparison against your real Cor/IoU numbers.

Run this AFTER score_with_official_toolkit.py, same folder, same setup.
"""

import json
import glob
import os
from pathlib import Path

from src import format_checker
from src import scorer as official_scorer

CACHE_DIR = r"D:\SHROOM\cache_outputs"
ORIGINALS_PATH = Path(r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl")
OUT_DIR = Path(r"D:\SHROOM\scoring_run")

POSSIBLE_LABELS = ["invention", "mischaracterization", "OCR", "miscounting", "other"]


def load_sample_ids_from_cache():
    """Same set of ids you actually scored -- for a fair comparison."""
    ids = set()
    search_path = os.path.join(CACHE_DIR, "cache_*.json")
    for file_path in glob.glob(search_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                ids.add(item["id"])
    return ids


def build_ref_dicts(originals, sample_ids):
    ref_dicts = []
    for sample_id in sample_ids:
        if sample_id not in originals:
            continue
        row = originals[sample_id]
        response_text = row.get("response", "")
        clean_gold = [
            {
                "start": int(s["start"]),
                "end": int(s["end"]),
                "prob": float(s.get("prob", 1.0)),
                "label": str(s["label"]),
            }
            for s in row.get("labels", [])
        ]
        ref_dicts.append({
            "id": sample_id,
            "labels": clean_gold,
            "text_len": len(response_text),
            "raw_annots": row.get("raw_annots", {}),
        })
    ref_dicts.sort(key=lambda r: r["id"])
    return ref_dicts


def build_baseline_preds(originals, sample_ids, baseline_label=None):
    """baseline_label=None -> mark_none (always predict empty).
    baseline_label='invention' etc -> flag the ENTIRE response as that label."""
    preds = []
    for sample_id in sample_ids:
        if sample_id not in originals:
            continue
        response_text = originals[sample_id].get("response", "")
        if baseline_label is None:
            labels = []
        else:
            labels = [{
                "start": 0,
                "end": len(response_text),
                "label": baseline_label,
                "prob": 1.0,
            }]
        preds.append({"id": sample_id, "labels": labels})
    preds.sort(key=lambda p: p["id"])
    return preds


def run_one_baseline(name, ref_dicts, pred_dicts):
    print(f"\n=== BASELINE: {name} ===")
    pred_ids = {p["id"] for p in pred_dicts}
    matched_refs = sorted([r for r in ref_dicts if r["id"] in pred_ids], key=lambda r: r["id"])
    assert len(matched_refs) == len(pred_dicts), f"mismatch: {len(matched_refs)} vs {len(pred_dicts)}"
    cors, cors_lbl = official_scorer.main(matched_refs, pred_dicts, output_file=None, verbose=True)


def main():
    sample_ids = load_sample_ids_from_cache()
    print(f"Comparing baselines on the same {len(sample_ids)} samples you already scored.\n")

    originals = {}
    with open(ORIGINALS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            originals[row["id"]] = row

    ref_dicts = build_ref_dicts(originals, sample_ids)

    # Baseline 1: always predict nothing
    preds = build_baseline_preds(originals, sample_ids, baseline_label=None)
    run_one_baseline("mark_none (always predict empty)", ref_dicts, preds)

    # Baseline 2-6: always flag the WHOLE response as each category
    for label in POSSIBLE_LABELS:
        preds = build_baseline_preds(originals, sample_ids, baseline_label=label)
        run_one_baseline(f"mark_{label} (flag entire response)", ref_dicts, preds)

    print("\n\nCompare these IoU/Cor numbers against your own model's score from "
          "score_with_official_toolkit.py. Your model should meaningfully beat all of these.")


if __name__ == "__main__":
    main()