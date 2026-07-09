"""
Generates the charts/tables you'll want for the system paper:
  1. Bar chart: average IoU per category (from all_results.json)
  2. Bar chart: sample count breakdown (zero/partial/perfect) per category
  3. Bar chart: your manual verdict breakdown (from manual_review_log.csv)
  4. A markdown summary table you can paste into your paper

Run this AFTER test_offline.py (needs all_results.json) and after you've
logged at least a few manual verdicts via view_sample.py (optional -- charts
1 and 2 work without it).
"""

import json
import csv
import os
from collections import defaultdict
import matplotlib.pyplot as plt

ALL_RESULTS_PATH = r"D:\SHROOM\analysis_outputs\all_results.json"
DATA_PATH = r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl"
REVIEW_LOG_PATH = r"D:\SHROOM\manual_review_log.csv"
OUTPUT_DIR = r"D:\SHROOM\paper_charts"

CATEGORY_ORDER = ["invention", "mischaracterization", "miscounting", "ocr", "other", "none"]


def load_category_for_id(data_path):
    cat_by_id = {}
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            labels = row.get("labels", [])
            cat_by_id[row["id"]] = labels[0]["label"].lower() if labels else "none"
    return cat_by_id


def chart_avg_iou_per_category():
    with open(ALL_RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)
    cat_by_id = load_category_for_id(DATA_PATH)

    by_cat = defaultdict(list)
    for r in results:
        cat = cat_by_id.get(r["id"], "other")
        by_cat[cat].append(r.get("iou", 0.0))

    cats = [c for c in CATEGORY_ORDER if c in by_cat]
    avgs = [sum(by_cat[c]) / len(by_cat[c]) for c in cats]
    counts = [len(by_cat[c]) for c in cats]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(cats, avgs, color="#4C72B0")
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"n={n}", ha="center", fontsize=9)
    ax.set_ylabel("Average IoU")
    ax.set_title("Average IoU by Hallucination Category")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "avg_iou_by_category.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    return cats, avgs, counts


def chart_iou_distribution_per_category():
    with open(ALL_RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)
    cat_by_id = load_category_for_id(DATA_PATH)

    by_cat = defaultdict(list)
    for r in results:
        cat = cat_by_id.get(r["id"], "other")
        by_cat[cat].append(r.get("iou", 0.0))

    cats = [c for c in CATEGORY_ORDER if c in by_cat]
    zero_pct, partial_pct, perfect_pct = [], [], []
    for c in cats:
        ious = by_cat[c]
        n = len(ious)
        zero_pct.append(100 * sum(1 for x in ious if x == 0.0) / n)
        perfect_pct.append(100 * sum(1 for x in ious if x == 1.0) / n)
        partial_pct.append(100 - zero_pct[-1] - perfect_pct[-1])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(cats, zero_pct, label="Zero IoU", color="#C44E52")
    ax.bar(cats, partial_pct, bottom=zero_pct, label="Partial IoU", color="#DD8452")
    bottom2 = [z + p for z, p in zip(zero_pct, partial_pct)]
    ax.bar(cats, perfect_pct, bottom=bottom2, label="Perfect IoU", color="#55A868")
    ax.set_ylabel("% of samples")
    ax.set_title("IoU Outcome Distribution by Category")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "iou_distribution_by_category.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def chart_manual_verdicts():
    if not os.path.exists(REVIEW_LOG_PATH):
        print("No manual_review_log.csv found yet -- skipping verdict chart.")
        return

    verdict_counts = defaultdict(int)
    with open(REVIEW_LOG_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            verdict_counts[row["verdict"]] += 1

    if not verdict_counts:
        print("manual_review_log.csv is empty -- skipping verdict chart.")
        return

    labels = list(verdict_counts.keys())
    values = [verdict_counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color="#8172B2")
    ax.set_ylabel("Count")
    ax.set_title(f"Manual Review Verdicts (n={sum(values)} reviewed)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "manual_verdict_breakdown.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

    print("\nVerdict breakdown:")
    for k, v in sorted(verdict_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({100*v/sum(values):.1f}%)")


def write_markdown_summary(cats, avgs, counts):
    out_path = os.path.join(OUTPUT_DIR, "summary_table.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("| Category | n | Average IoU |\n")
        f.write("|---|---|---|\n")
        for c, a, n in zip(cats, avgs, counts):
            f.write(f"| {c} | {n} | {a:.3f} |\n")
    print(f"Saved: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cats, avgs, counts = chart_avg_iou_per_category()
    chart_iou_distribution_per_category()
    chart_manual_verdicts()
    write_markdown_summary(cats, avgs, counts)
    print(f"\nAll charts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()