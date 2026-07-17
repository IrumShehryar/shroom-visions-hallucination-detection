"""
Confidence-threshold sweep: cheap offline post-processing search.

Reuses the already-cached scoring_run/{predictions,references}.jsonl (no
new API calls) to test whether dropping low-confidence predicted spans --
globally, and specifically for the "mischaracterization" label, which the
error-analysis outputs flagged as the worst-performing and most-frequent
false-positive source -- improves Cor / Cor+Lbl against the official
scorer.
"""
import json
from pathlib import Path

from src import scorer as official_scorer

PRED_PATH = Path(r"D:\SHROOM\scoring_run\predictions.jsonl")
REF_PATH = Path(r"D:\SHROOM\scoring_run\references.jsonl")

GLOBAL_THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
MISCHAR_THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def filter_labels(pred_dicts, global_thresh, mischar_thresh=None):
    out = []
    for row in pred_dicts:
        labs = []
        for l in row["labels"]:
            thresh = (
                mischar_thresh
                if (mischar_thresh is not None and l["label"] == "mischaracterization")
                else global_thresh
            )
            if l["prob"] >= thresh:
                labs.append(l)
        out.append({"id": row["id"], "labels": labs})
    return out


def score(ref_dicts, pred_dicts):
    ref_dicts_sorted = sorted(ref_dicts, key=lambda d: d["id"])
    pred_dicts_sorted = sorted(pred_dicts, key=lambda d: d["id"])
    ids_r = [d["id"] for d in ref_dicts_sorted]
    ids_p = [d["id"] for d in pred_dicts_sorted]
    assert ids_r == ids_p, "id mismatch between refs and preds"

    cors = [official_scorer.score_cor(r, p) for r, p in zip(ref_dicts_sorted, pred_dicts_sorted)]
    cors_lbl = [official_scorer.score_cor_lbl(r, p) for r, p in zip(ref_dicts_sorted, pred_dicts_sorted)]
    ious = [official_scorer.score_iou(r, p) for r, p in zip(ref_dicts_sorted, pred_dicts_sorted)]
    n = len(cors)
    return sum(cors) / n, sum(cors_lbl) / n, sum(ious) / n


def main():
    raw_preds = load(PRED_PATH)
    raw_refs = load(REF_PATH)

    pred_ids = {d["id"] for d in raw_preds}
    raw_refs = [d for d in raw_refs if d["id"] in pred_ids]
    print(f"matched samples: {len(raw_refs)}")

    header = f"{'global_thresh':<14}{'mischar_thresh':<16}{'Cor':<10}{'Cor+Lbl':<10}{'IoU':<10}"
    print(header)

    cor, cl, iou = score(raw_refs, raw_preds)
    print(f"{'baseline(0.0)':<14}{'--':<16}{cor:<10.4f}{cl:<10.4f}{iou:<10.4f}")

    print("\n-- global threshold sweep (applies to ALL labels) --")
    for t in GLOBAL_THRESHOLDS:
        filtered = filter_labels(raw_preds, t)
        cor, cl, iou = score(raw_refs, filtered)
        print(f"{t:<14}{'--':<16}{cor:<10.4f}{cl:<10.4f}{iou:<10.4f}")

    print("\n-- mischaracterization-only threshold sweep (all other labels unfiltered) --")
    for t in MISCHAR_THRESHOLDS:
        filtered = filter_labels(raw_preds, 0.0, mischar_thresh=t)
        cor, cl, iou = score(raw_refs, filtered)
        print(f"{'0.0':<14}{t:<16}{cor:<10.4f}{cl:<10.4f}{iou:<10.4f}")


if __name__ == "__main__":
    main()
