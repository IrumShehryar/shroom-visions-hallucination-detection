import json
from src.verify_flag_with_sonnet import call_sonnet_verify
import os

IMAGES_DIR = r"D:\SHROOM\distrib\images\shroom-vis-images"


def run(cands, kind, originals, raw_cache):
    out = []
    for sid, span, prob in cands:
        row = originals[sid]
        item = raw_cache[sid]
        raw = item.get("haiku_reasoning", "[]").replace("```json", "").replace("```", "").strip()
        try:
            llm_out = json.loads(raw)
        except json.JSONDecodeError:
            llm_out = []
        entry = next((e for e in llm_out if e.get("span_text", "").strip() == span.strip()), None)
        if entry is None:
            entry = next((e for e in llm_out if span.strip() in e.get("span_text", "") or e.get("span_text","").strip() in span.strip()), None)
        if entry is None:
            print(f"[{kind}] {sid}: could not locate span {span!r}, skipping")
            continue
        image_path = os.path.join(IMAGES_DIR, row.get("image_name", ""))
        result_text, in_tok, out_tok = call_sonnet_verify(
            image_path, row["response"], span, entry.get("label", "mischaracterization"), entry.get("reason", "")
        )
        try:
            clean = result_text.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            verdict_obj = json.loads(clean.strip())
        except Exception:
            verdict_obj = {"verdict": "PARSE_ERROR", "confidence": None, "reason": result_text[:200]}
        print(f"[{kind}] {sid:<16} span={span[:40]!r:<45} -> {verdict_obj.get('verdict')} ({verdict_obj.get('confidence')})  in={in_tok} out={out_tok}")
        out.append({"id": sid, "span": span, "verdict": verdict_obj, "in_tok": in_tok, "out_tok": out_tok})
    return out


def main():
    originals = {}
    for line in open(r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl", encoding="utf-8"):
        r = json.loads(line)
        originals[r["id"]] = r
    import glob
    raw_cache = {}
    for path in glob.glob("cache_outputs/cache_*.json"):
        for item in json.load(open(path, encoding="utf-8")):
            raw_cache[item["id"]] = item
    for item in json.load(open("fresh_validation_raw_cache.json", encoding="utf-8")):
        raw_cache[item["id"]] = item

    data = json.load(open("scratchpad_dev_final.json", encoding="utf-8"))
    fp_results = run(data["keep_fp"], "FP", originals, raw_cache)
    tp_results = run(data["keep_tp"], "TP", originals, raw_cache)

    with open("verify_dev_expand_results.json", "w", encoding="utf-8") as f:
        json.dump({"fp": fp_results, "tp": tp_results}, f, indent=2, ensure_ascii=False)

    total_in = sum(r["in_tok"] for r in fp_results + tp_results)
    total_out = sum(r["out_tok"] for r in fp_results + tp_results)
    cost = total_in * 2.00 / 1e6 + total_out * 10.00 / 1e6
    print(f"\nDev expansion: {len(fp_results)} FP + {len(tp_results)} TP = {len(fp_results)+len(tp_results)} calls, cost=${cost:.4f}")


if __name__ == "__main__":
    main()
