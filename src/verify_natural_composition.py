"""
Confirms the curated 51-sample Sonnet-verification result on a genuinely
NATURAL-composition sample: instead of hand-picking a ~50/50 FP/TP mix,
draw a plain random sample from every mischaracterization flag the current
pipeline actually produces (pred_fix, post numeral-relabel), in whatever
ratio they naturally occur in. Then run the same targeted Sonnet
verification and apply the real scorer before/after on the touched rows.

Already-verified (id, span_text) pairs from the three curated batches are
excluded so this is genuinely new evidence.
"""
import json
import glob
import os
import random

from src.relabel_miscounting import relabel_miscounting
from src.span_mapper import normalize_text, isolate_span, _find_unused_occurrence
from src.confusion_matrix_analysis import overlaps
from src.verify_flag_with_sonnet import call_sonnet_verify
from src import scorer as official_scorer

TRAIN_PATH = r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl"
IMAGES_DIR = r"D:\SHROOM\distrib\images\shroom-vis-images"
SEED = 2024
N_SAMPLE = 60
OUT_PATH = r"D:\SHROOM\verify_natural_composition_results.json"


def already_verified_verdicts():
    """(id, span_text) -> verdict dict, pooled from the three curated batches,
    so a natural-composition draw that happens to re-include one of these
    reuses the real verdict instead of re-paying for a (non-deterministic,
    see fix_mischaracterization confidence notes) repeat call."""
    verdicts = {}
    for fname in ("verify_batch_results.json", "verify_heldout_results.json"):
        data = json.load(open(fname, encoding="utf-8"))
        for kind, items in data.items():
            for r in items:
                verdicts[(r["id"], r["span"].strip())] = r["verdict"]
    data = json.load(open("verify_dev_expand_results.json", encoding="utf-8"))
    for kind, items in data.items():
        for r in items:
            verdicts[(r["id"], r["span"].strip())] = r["verdict"]
    return verdicts


def build_pool():
    originals = {}
    for line in open(TRAIN_PATH, encoding="utf-8"):
        r = json.loads(line)
        originals[r["id"]] = r

    sources = []  # (id, raw_llm_reasoning, gold_labels, response)
    for path in glob.glob("cache_outputs/cache_*.json"):
        for item in json.load(open(path, encoding="utf-8")):
            row = originals.get(item["id"])
            if not row:
                continue
            sources.append((item["id"], item.get("haiku_reasoning", "[]"), row.get("labels", []), row["response"]))
    for item in json.load(open("fresh_validation_raw_cache.json", encoding="utf-8")):
        row = originals.get(item["id"], {})
        sources.append((item["id"], item.get("haiku_reasoning", "[]"), item.get("gold_labels", []), row.get("response", "")))

    pool = []
    for sid, raw, gold, response in sources:
        raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            llm_out = json.loads(raw)
        except json.JSONDecodeError:
            continue
        llm_out = relabel_miscounting(llm_out)
        gold_mischar = [g for g in gold if g.get("label") == "mischaracterization"]

        normalized_response = normalize_text(response)
        already_used = []
        for entry in llm_out:
            raw_span = normalize_text(entry.get("span_text", "")).strip()
            label = str(entry.get("label", "other")).lower()
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
            if label != "mischaracterization":
                continue
            pred_span = {"start": idx, "end": end, "label": "mischaracterization"}
            is_tp = any(overlaps(pred_span, g) for g in gold_mischar)
            pool.append({
                "id": sid, "span_text": raw_span, "start": idx, "end": end,
                "reason": entry.get("reason", ""), "prob": entry.get("prob", 0.5),
                "is_tp": is_tp, "response": response, "gold": gold,
            })
    return pool


def main():
    pool = build_pool()
    tp_n = sum(1 for p in pool if p["is_tp"])
    fp_n = len(pool) - tp_n
    print(f"Natural pool: {len(pool)} mischaracterization flags (current pipeline, post numeral-fix)")
    print(f"  TP (overlaps gold mischaracterization): {tp_n} ({tp_n/len(pool):.1%})")
    print(f"  FP (no gold mischaracterization overlap): {fp_n} ({fp_n/len(pool):.1%})")

    reuse = already_verified_verdicts()
    print(f"{len(reuse)} (id, span) pairs already have a Sonnet verdict from prior curated batches -- reused, not re-called")

    random.seed(SEED)
    random.shuffle(pool)
    sample = pool[:N_SAMPLE]
    sample_tp = sum(1 for p in sample if p["is_tp"])
    n_reused = sum(1 for p in sample if (p["id"], p["span_text"]) in reuse)
    print(f"Drawn sample (from FULL natural pool, true composition preserved): {len(sample)} flags -- "
          f"TP={sample_tp} ({sample_tp/len(sample):.1%})  FP={len(sample)-sample_tp} ({(len(sample)-sample_tp)/len(sample):.1%})")
    print(f"  of which {n_reused} reuse a cached verdict, {len(sample)-n_reused} require a fresh Sonnet call")

    originals = {}
    for line in open(TRAIN_PATH, encoding="utf-8"):
        r = json.loads(line)
        originals[r["id"]] = r

    results = []
    total_in, total_out = 0, 0
    for i, item in enumerate(sample, 1):
        key = (item["id"], item["span_text"])
        kind = "TP" if item["is_tp"] else "FP"
        if key in reuse:
            verdict_obj = reuse[key]
            print(f"[{i}/{len(sample)}] [{kind}] {item['id']:<16} span={item['span_text'][:40]!r:<45} -> {verdict_obj.get('verdict')} ({verdict_obj.get('confidence')})  [reused]")
            results.append({
                "id": item["id"], "span": item["span_text"], "start": item["start"], "end": item["end"],
                "is_tp": item["is_tp"], "gold": item["gold"], "verdict": verdict_obj,
            })
            continue
        row = originals.get(item["id"], {})
        image_name = row.get("image_name", "")
        image_path = os.path.join(IMAGES_DIR, image_name)
        if not os.path.exists(image_path):
            print(f"[{i}/{len(sample)}] {item['id']}: image missing, skipping")
            continue
        result_text, in_tok, out_tok = call_sonnet_verify(
            image_path, item["response"], item["span_text"], "mischaracterization", item["reason"]
        )
        total_in += in_tok
        total_out += out_tok
        clean = result_text.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        try:
            verdict_obj = json.loads(clean.strip())
        except Exception:
            verdict_obj = {"verdict": "PARSE_ERROR", "confidence": None, "reason": result_text[:200]}
        print(f"[{i}/{len(sample)}] [{kind}] {item['id']:<16} span={item['span_text'][:40]!r:<45} -> {verdict_obj.get('verdict')} ({verdict_obj.get('confidence')})")
        results.append({
            "id": item["id"], "span": item["span_text"], "start": item["start"], "end": item["end"],
            "is_tp": item["is_tp"], "gold": item["gold"], "verdict": verdict_obj,
        })

    json.dump(results, open(OUT_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    fp_results = [r for r in results if not r["is_tp"]]
    tp_results = [r for r in results if r["is_tp"]]
    fp_reject = [r for r in fp_results if r["verdict"].get("verdict") == "reject"]
    tp_reject = [r for r in tp_results if r["verdict"].get("verdict") == "reject"]

    print(f"\n{'='*70}")
    if fp_results:
        print(f"NATURAL-SAMPLE catch rate (FP correctly rejected): {len(fp_reject)}/{len(fp_results)} = {len(fp_reject)/len(fp_results):.1%}")
    if tp_results:
        print(f"NATURAL-SAMPLE FN rate (TP wrongly rejected): {len(tp_reject)}/{len(tp_results)} = {len(tp_reject)/len(tp_results):.1%}")
    cost = total_in * 2.00 / 1e6 + total_out * 10.00 / 1e6
    print(f"Total usage: input={total_in} output={total_out}, cost=${cost:.4f}")

    # --- apply the real scorer, before vs after the drop-on-reject filter, on the touched rows ---
    by_id = {}
    for r in results:
        by_id.setdefault(r["id"], []).append(r)

    ref_dicts, pred_before, pred_after = [], [], []
    for sid, entries in by_id.items():
        row = originals[sid]
        response = row["response"]
        gold = entries[0]["gold"]
        # rebuild this row's full current-pipeline mischaracterization prediction set
        # (only the touched spans need before/after variants; other labels' spans
        # are identical in both, so we only need mischaracterization spans here
        # since that's the only label this filter can change)
        before_spans = [{"start": e["start"], "end": e["end"], "label": "mischaracterization", "prob": 0.8} for e in entries]
        after_spans = [{"start": e["start"], "end": e["end"], "label": "mischaracterization", "prob": 0.8}
                       for e in entries if e["verdict"].get("verdict") != "reject"]
        ref_dicts.append({"id": sid, "labels": gold, "text_len": len(response)})
        pred_before.append({"id": sid, "labels": before_spans})
        pred_after.append({"id": sid, "labels": after_spans})

    ref_sorted = sorted(ref_dicts, key=lambda d: d["id"])
    before_sorted = sorted(pred_before, key=lambda d: d["id"])
    after_sorted = sorted(pred_after, key=lambda d: d["id"])

    cor_before = sum(official_scorer.score_cor(r, p) for r, p in zip(ref_sorted, before_sorted)) / len(ref_sorted)
    cor_after = sum(official_scorer.score_cor(r, p) for r, p in zip(ref_sorted, after_sorted)) / len(ref_sorted)
    corlbl_before = sum(official_scorer.score_cor_lbl(r, p) for r, p in zip(ref_sorted, before_sorted)) / len(ref_sorted)
    corlbl_after = sum(official_scorer.score_cor_lbl(r, p) for r, p in zip(ref_sorted, after_sorted)) / len(ref_sorted)

    print(f"\nSCORER on {len(ref_sorted)} touched rows (mischaracterization spans only, isolated from other labels):")
    print(f"  Cor:      before={cor_before:.4f}  after={cor_after:.4f}  delta={cor_after-cor_before:+.4f}")
    print(f"  Cor+Lbl:  before={corlbl_before:.4f}  after={corlbl_after:.4f}  delta={corlbl_after-corlbl_before:+.4f}")


if __name__ == "__main__":
    main()
