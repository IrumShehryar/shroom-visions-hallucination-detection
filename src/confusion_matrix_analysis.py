"""
Systematic category-confusion analysis across the combined 286+226+39
labeled sample pool (scoring_run + non-overlapping old_prompt_v1_scoring_run
+ fresh_validation_raw_cache), using the CURRENT pipeline (span_mapper.py,
now including the merged numeral-relabeling fix).

Produces:
  1. A 5x5 gold-label x predicted-label confusion matrix (any char overlap).
  2. Per-label Cor+Lbl sub-scores (score_cor with label_filtered_ fixed per
     label, averaged across all rows -- not just rows where that label
     appears, unlike the official per-row union average).
  3. A detection-vs-labeling error split per gold category.
  4. A with-fix vs without-fix comparison isolating the effect of the
     numeral relabeling rule on the miscounting<->mischaracterization cells,
     plus a scan for any other large off-diagonal cell.
"""
import json
import glob
from collections import defaultdict

from src import scorer as official_scorer
from src.span_mapper import map_spans_to_characters, isolate_span, normalize_text, _find_unused_occurrence
from src.relabel_miscounting import _is_excluded
from src.span_mapper import NUMBER_WORDS, _tokenize_words

LABELS = ["invention", "mischaracterization", "miscounting", "OCR", "other"]
TRAIN_PATH = r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl"


def map_spans_no_fix(llm_output, response_text):
    """Replica of map_spans_to_characters WITHOUT the relabel step, for the
    with-fix vs without-fix comparison in part 4."""
    mapped = []
    already_used = []
    normalized_response = normalize_text(response_text)
    LABEL_FIX = {
        "invention": "invention", "mischaracterization": "mischaracterization",
        "ocr": "OCR", "miscounting": "miscounting", "other": "other",
    }
    for item in llm_output:
        raw_span = normalize_text(item.get("span_text", "")).strip()
        raw_label = item.get("label", "other")
        label = LABEL_FIX.get(str(raw_label).lower(), str(raw_label).lower())
        prob = item.get("prob", 0.5)
        if not raw_span:
            continue
        target = isolate_span(raw_span, label, normalized_response)
        if not target:
            continue
        idx = _find_unused_occurrence(normalized_response, target, already_used)
        if idx == -1:
            idx = _find_unused_occurrence(normalized_response.lower(), target.lower(), already_used)
        if idx == -1:
            continue
        end = idx + len(target)
        already_used.append((idx, end))
        mapped.append({"start": idx, "end": end, "label": label, "prob": round(float(prob), 4)})
    mapped.sort(key=lambda x: x["start"])
    return mapped


def relabel_mapped_span(response, l):
    """Apply the fix directly to an already-mapped span (for the 226-set,
    where only final mapped predictions are available, not raw output)."""
    l = dict(l)
    if l["label"] != "mischaracterization":
        return l
    span_text = response[l["start"]:l["end"]]
    toks = _tokenize_words(span_text)
    has_num = any(t.isdigit() or t.lower() in NUMBER_WORDS for t in toks)
    if not (has_num and 0 < len(toks) <= 3 and not _is_excluded(span_text, toks)):
        return l
    for t in toks:
        if t.isdigit() or t.lower() in NUMBER_WORDS:
            idx = response.find(t, l["start"], l["end"])
            if idx != -1:
                return {"start": idx, "end": idx + len(t), "prob": l["prob"], "label": "miscounting"}
    return l


def load_combined():
    """Returns list of {id, response, gold, pred_fix, pred_nofix}."""
    originals = {}
    for line in open(TRAIN_PATH, encoding="utf-8"):
        r = json.loads(line)
        originals[r["id"]] = r

    rows = []

    # --- 286-sample set (raw haiku output available) ---
    dev_items = []
    for path in glob.glob("cache_outputs/cache_*.json"):
        dev_items.extend(json.load(open(path, encoding="utf-8")))
    for item in dev_items:
        row = originals.get(item["id"])
        if not row:
            continue
        response = row["response"]
        raw = item.get("haiku_reasoning", "[]").replace("```json", "").replace("```", "").strip()
        try:
            llm_out = json.loads(raw)
        except json.JSONDecodeError:
            llm_out = []
        rows.append({
            "id": item["id"], "response": response, "gold": row.get("labels", []),
            "pred_fix": map_spans_to_characters(llm_out, response),
            "pred_nofix": map_spans_no_fix(llm_out, response),
        })

    # --- 39-sample fresh held-out set (raw haiku output available) ---
    fresh_items = json.load(open("fresh_validation_raw_cache.json", encoding="utf-8"))
    for item in fresh_items:
        row = originals.get(item["id"], {})
        response = row.get("response", "")
        raw = item.get("haiku_reasoning", "[]").replace("```json", "").replace("```", "").strip()
        try:
            llm_out = json.loads(raw)
        except json.JSONDecodeError:
            llm_out = []
        rows.append({
            "id": item["id"], "response": response, "gold": item.get("gold_labels", []),
            "pred_fix": map_spans_to_characters(llm_out, response),
            "pred_nofix": map_spans_no_fix(llm_out, response),
        })

    # --- 226-sample non-overlapping old-prompt set (only mapped preds available) ---
    p344 = [json.loads(l) for l in open("old_prompt_v1_scoring_run/predictions.jsonl", encoding="utf-8")]
    r344 = [json.loads(l) for l in open("old_prompt_v1_scoring_run/references.jsonl", encoding="utf-8")]
    r344_by_id = {r["id"]: r for r in r344}
    p286_ids = {json.loads(l)["id"] for l in open("scoring_run/predictions.jsonl", encoding="utf-8")}
    for p in p344:
        if p["id"] in p286_ids:
            continue
        ref = r344_by_id.get(p["id"])
        if not ref:
            continue
        response = ref["response"]
        pred_nofix = p["labels"]
        pred_fix = [relabel_mapped_span(response, l) for l in pred_nofix]
        rows.append({
            "id": p["id"], "response": response, "gold": ref["labels"],
            "pred_fix": pred_fix, "pred_nofix": pred_nofix,
        })

    return rows


def overlaps(a, b):
    return not (a["end"] <= b["start"] or a["start"] >= b["end"])


def build_confusion(rows, pred_key):
    matrix = defaultdict(lambda: defaultdict(int))
    missed = defaultdict(int)
    correct = defaultdict(int)
    wrong = defaultdict(int)
    gold_totals = defaultdict(int)
    unmatched_preds = defaultdict(int)  # predicted spans with no overlapping gold at all

    for row in rows:
        gold_spans = row["gold"]
        pred_spans = row[pred_key]
        gold_covered_by_pred = [False] * len(gold_spans)

        for gi, g in enumerate(gold_spans):
            gold_totals[g["label"]] += 1
            overlapping = [p for p in pred_spans if overlaps(g, p)]
            if not overlapping:
                missed[g["label"]] += 1
                matrix[g["label"]]["MISSED"] += 1
                continue
            overlapping_labels = {p["label"] for p in overlapping}
            if g["label"] in overlapping_labels:
                correct[g["label"]] += 1
                matrix[g["label"]][g["label"]] += 1
            else:
                # pick the highest-confidence overlapping wrong-label prediction
                best = max(overlapping, key=lambda p: p.get("prob", 0))
                wrong[g["label"]] += 1
                matrix[g["label"]][best["label"]] += 1

        for p in pred_spans:
            if not any(overlaps(g, p) for g in gold_spans):
                unmatched_preds[p["label"]] += 1

    return matrix, gold_totals, missed, correct, wrong, unmatched_preds


def print_confusion_matrix(matrix, gold_totals, title):
    print(f"\n{title}")
    cols = LABELS + ["MISSED"]
    row_label = "gold / pred"
    header = f"{row_label:<20}" + "".join(f"{c:>16}" for c in cols)
    print(header)
    for g in LABELS:
        row_vals = [matrix[g].get(c, 0) for c in cols]
        total = gold_totals[g]
        pct_row = [f"{v} ({v/total:.0%})" if total else f"{v}" for v in row_vals]
        print(f"{g:<20}" + "".join(f"{v:>16}" for v in pct_row))
    print(f"{'(gold totals)':<20}" + "".join(f"{gold_totals[g]:>16}" for g in LABELS) + f"{'':>16}")


def per_label_cor_sub_scores(rows, pred_key):
    ref_dicts = [{"id": r["id"], "labels": r["gold"], "text_len": len(r["response"])} for r in rows]
    pred_dicts = [{"id": r["id"], "labels": r[pred_key]} for r in rows]
    ref_sorted = sorted(ref_dicts, key=lambda d: d["id"])
    pred_sorted = sorted(pred_dicts, key=lambda d: d["id"])
    scores = {}
    for label in LABELS:
        vals = [official_scorer.score_cor(r, p, label_filtered_=label) for r, p in zip(ref_sorted, pred_sorted)]
        scores[label] = sum(vals) / len(vals)
    return scores


def main():
    rows = load_combined()
    print(f"Combined dataset: {len(rows)} samples (286 dev + 226 old-prompt + 39 held-out)")

    # --- Part 1: confusion matrix (with fix, i.e. current pipeline) ---
    matrix_fix, gold_totals, missed, correct, wrong, unmatched = build_confusion(rows, "pred_fix")
    print_confusion_matrix(matrix_fix, gold_totals, "PART 1: CONFUSION MATRIX (current pipeline, with numeral fix)")
    print("\nUnmatched predictions (flagged, no overlapping gold span at all -- pure false positives):")
    for lab in LABELS:
        print(f"  {lab:<20} {unmatched.get(lab, 0)}")

    # --- Part 2: per-label Cor+Lbl sub-scores ---
    scores_fix = per_label_cor_sub_scores(rows, "pred_fix")
    print("\nPART 2: PER-LABEL Cor SUB-SCORE (label_filtered_ fixed per label, averaged over all rows)")
    for label in LABELS:
        print(f"  {label:<20} {scores_fix[label]:.4f}")
    overall_cor_lbl_style = sum(scores_fix.values()) / len(scores_fix)
    print(f"  {'(simple mean)':<20} {overall_cor_lbl_style:.4f}  [not the official per-row Cor+Lbl -- see note]")

    # --- Part 3: detection vs labeling split ---
    print("\nPART 3: DETECTION vs LABELING ERROR SPLIT (per gold category)")
    print(f"{'label':<20}{'missed %':>12}{'wrong-label %':>16}{'correct %':>12}{'n':>8}")
    for label in LABELS:
        n = gold_totals[label]
        if n == 0:
            continue
        print(f"{label:<20}{missed[label]/n:>11.1%}{wrong[label]/n:>15.1%}{correct[label]/n:>11.1%}{n:>8}")

    # --- Part 4: with-fix vs without-fix comparison ---
    matrix_nofix, gold_totals_nofix, missed_nf, correct_nf, wrong_nf, unmatched_nf = build_confusion(rows, "pred_nofix")
    print_confusion_matrix(matrix_nofix, gold_totals_nofix, "PART 4a: CONFUSION MATRIX (WITHOUT the numeral fix, for comparison)")

    print("\nPART 4b: target cell before/after the fix")
    print(f"  miscounting gold -> predicted mischaracterization:  before={matrix_nofix['miscounting'].get('mischaracterization',0)}  after={matrix_fix['miscounting'].get('mischaracterization',0)}")
    print(f"  mischaracterization gold -> predicted miscounting:  before={matrix_nofix['mischaracterization'].get('miscounting',0)}  after={matrix_fix['mischaracterization'].get('miscounting',0)}")

    print("\nPART 4c: other off-diagonal cells, ranked by count (with-fix matrix, MISSED excluded)")
    off_diag = []
    for g in LABELS:
        for p in LABELS:
            if g == p:
                continue
            v = matrix_fix[g].get(p, 0)
            if v > 0:
                off_diag.append((v, g, p))
    off_diag.sort(reverse=True)
    for v, g, p in off_diag:
        print(f"  gold={g:<20} -> pred={p:<20} count={v}  ({v/gold_totals[g]:.1%} of gold {g})")


if __name__ == "__main__":
    main()
