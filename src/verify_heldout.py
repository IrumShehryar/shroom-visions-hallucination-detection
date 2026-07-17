import json
import os
from src.verify_flag_with_sonnet import call_sonnet_verify

IMAGES_DIR = r"D:\SHROOM\distrib\images\shroom-vis-images"

MAIN_BATCH = {
    "FP": [
        ("train-en-477", "lighter, almost white or lavender stripes"),
        ("train-en-2371", "cracked asphalt or concrete"),
        ("train-en-2057", "propellers"),
        ("train-en-1828", "The yellow and teal containers hold whole coconut fruits that still possess their hard outer shell"),
        ("train-en-11101", "blue SUV"),
        ("train-en-830", "bright"),
        ("train-en-469", "does not appear to have stitching"),
    ],
    "TP": [
        ("train-en-876", "still has its skin on"),
        ("train-en-1483", "rectangular blade"),
        ("train-en-11139", "rose or a floral pattern"),
        ("train-en-2207", "grayish areas around its neck"),
        ("train-en-1727", "suspended from the ceiling by chains or straps"),
        ("train-en-9671", "blue and red text"),
    ],
}


def main():
    originals = {}
    for line in open(r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl", encoding="utf-8"):
        r = json.loads(line)
        originals[r["id"]] = r
    fresh = {item["id"]: item for item in json.load(open("fresh_verify_raw_cache.json", encoding="utf-8"))}

    results = {"FP": [], "TP": []}
    total_in, total_out = 0, 0

    for kind, items in MAIN_BATCH.items():
        for sid, span in items:
            row = originals[sid]
            item = fresh[sid]
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
            total_in += in_tok
            total_out += out_tok
            try:
                clean = result_text.strip()
                if clean.startswith("```"):
                    clean = clean.split("```")[1]
                    if clean.startswith("json"):
                        clean = clean[4:]
                verdict_obj = json.loads(clean.strip())
            except Exception:
                verdict_obj = {"verdict": "PARSE_ERROR", "confidence": None, "reason": result_text[:200]}
            print(f"[{kind}] {sid:<16} span={span[:40]!r:<45} -> {verdict_obj.get('verdict')} ({verdict_obj.get('confidence')})")
            results[kind].append({"id": sid, "span": span, "verdict": verdict_obj})

    with open("verify_heldout_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    fp_results, tp_results = results["FP"], results["TP"]
    fp_reject = [r for r in fp_results if r["verdict"].get("verdict") == "reject"]
    fp_confirm = [r for r in fp_results if r["verdict"].get("verdict") == "confirm"]
    tp_confirm = [r for r in tp_results if r["verdict"].get("verdict") == "confirm"]
    tp_reject = [r for r in tp_results if r["verdict"].get("verdict") == "reject"]

    n_fp, n_tp = len(fp_reject)+len(fp_confirm), len(tp_confirm)+len(tp_reject)
    if n_fp:
        print(f"HELD-OUT catch rate: {len(fp_reject)}/{n_fp} = {len(fp_reject)/n_fp:.1%}")
    if n_tp:
        print(f"HELD-OUT FN rate: {len(tp_reject)}/{n_tp} = {len(tp_reject)/n_tp:.1%}")

    cost = total_in * 2.00 / 1e6 + total_out * 10.00 / 1e6
    print(f"Total usage: input={total_in} output={total_out}, cost=${cost:.4f}")


if __name__ == "__main__":
    main()
