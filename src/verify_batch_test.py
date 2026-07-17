"""
Second-opinion Sonnet verification test, scoped to mischaracterization
flags only, on a properly-sized batch (17: 9 known false positives + 8
known true positives). 5 additional candidates were screened out as
confounded (small image + fine-grained discrimination content -- species,
product-type, color, material) and are reported separately, not folded
into the main batch.
"""
import json
import glob
import os
from src.verify_flag_with_sonnet import call_sonnet_verify

IMAGES_DIR = r"D:\SHROOM\distrib\images\shroom-vis-images"

MAIN_BATCH = {
    "FP": [
        ("train-en-2657", "There are no words in green text in the image."),
        ("train-en-2037", "not composed of circular tubes"),
        ("train-en-1766", "white stripes on its back"),
        ("train-en-11473", "The wings and upper body are a mix of dark brown and grayish-brown"),
        ("train-en-11409", "paraglider wing"),
        ("train-en-992", "green flooring"),
        ("train-en-1693", "bralette"),
        ("train-en-1478", "gold ring"),
        ("train-en-11050", "Pumpkins are usually green or orange"),
    ],
    "TP": [
        ("train-en-11357", "It extends from the shoulder to the elbow area, covering the upper arm"),
        ("train-en-11736", "circular base that is not visible from the current angle"),
        ("train-en-920", "long barrel"),
        ("train-en-1959", "multiple layers of thick brown leather"),
        ("train-en-11886", "primarily black"),
        ("train-en-966", "rectangular blade"),
        ("train-en-11402", "This is not an orange, but rather a grapefruit."),
        ("train-en-11218", "two small front wheels connected by a frame bar"),
    ],
}

CONFOUNDED = [
    ("train-en-2271", "FP", "strawberry", "800x700, 91936 bytes -- species/variety discrimination"),
    ("train-en-1602", "TP", "crayfish", "1000x616, 103464 bytes -- species discrimination"),
    ("train-en-560", "TP", "This is a modern compound crossbow", "1000x667, 125152 bytes -- product-type discrimination"),
    ("train-en-11472", "TP", "black", "700x969, 125360 bytes -- color discrimination"),
    ("train-en-9860", "TP", "electric engine", "1000x750, 128914 bytes -- material/mechanism discrimination"),
]


def main():
    originals = {}
    for line in open(r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl", encoding="utf-8"):
        r = json.loads(line)
        originals[r["id"]] = r

    raw_cache = {}
    for path in glob.glob("cache_outputs/cache_*.json"):
        for item in json.load(open(path, encoding="utf-8")):
            raw_cache[item["id"]] = item
    for item in json.load(open("fresh_validation_raw_cache.json", encoding="utf-8")):
        raw_cache[item["id"]] = item

    results = {"FP": [], "TP": []}
    total_in, total_out = 0, 0

    for kind, items in MAIN_BATCH.items():
        for sid, target_span in items:
            row = originals[sid]
            item = raw_cache[sid]
            raw = item.get("haiku_reasoning", "[]").replace("```json", "").replace("```", "").strip()
            try:
                llm_out = json.loads(raw)
            except json.JSONDecodeError:
                llm_out = []
            entry = next((e for e in llm_out if e.get("span_text", "").strip() == target_span.strip()), None)
            if entry is None:
                # fallback: substring match
                entry = next((e for e in llm_out if target_span.strip() in e.get("span_text", "") or e.get("span_text","").strip() in target_span.strip()), None)
            if entry is None:
                print(f"[{kind}] {sid}: could not locate span {target_span!r}, skipping")
                continue

            image_path = os.path.join(IMAGES_DIR, row.get("image_name", ""))
            result_text, in_tok, out_tok = call_sonnet_verify(
                image_path, row["response"], target_span, entry.get("label", "mischaracterization"), entry.get("reason", "")
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

            print(f"[{kind}] {sid:<16} span={target_span[:40]!r:<45} -> {verdict_obj.get('verdict')} ({verdict_obj.get('confidence')})")
            results[kind].append({"id": sid, "span": target_span, "verdict": verdict_obj})

    with open("verify_batch_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # --- metrics ---
    print("\n" + "=" * 70)
    fp_results = results["FP"]
    tp_results = results["TP"]

    fp_rejected = [r for r in fp_results if r["verdict"].get("verdict") == "reject"]
    fp_confirmed = [r for r in fp_results if r["verdict"].get("verdict") == "confirm"]
    tp_confirmed = [r for r in tp_results if r["verdict"].get("verdict") == "confirm"]
    tp_rejected = [r for r in tp_results if r["verdict"].get("verdict") == "reject"]

    n_fp, n_tp = len(fp_results), len(tp_results)
    print(f"Catch rate (known FPs correctly rejected): {len(fp_rejected)}/{n_fp} = {len(fp_rejected)/n_fp:.1%}" if n_fp else "no FP results")
    print(f"False-negative-on-TPs rate (known TPs wrongly rejected): {len(tp_rejected)}/{n_tp} = {len(tp_rejected)/n_tp:.1%}" if n_tp else "no TP results")

    def confs(lst):
        return [r["verdict"].get("confidence") for r in lst if isinstance(r["verdict"].get("confidence"), (int, float))]

    import statistics as st
    for label, lst in [("FP correctly rejected", fp_rejected), ("FP wrongly confirmed", fp_confirmed),
                        ("TP correctly confirmed", tp_confirmed), ("TP wrongly rejected", tp_rejected)]:
        c = confs(lst)
        if c:
            print(f"  {label}: n={len(c)} mean_conf={st.mean(c):.2f} values={c}")
        else:
            print(f"  {label}: n=0")

    print(f"\nTotal usage: input={total_in} output={total_out} across {len(fp_results)+len(tp_results)} calls")
    cost = total_in * 2.00 / 1e6 + total_out * 10.00 / 1e6
    print(f"Actual cost: ${cost:.4f}")

    print("\n" + "=" * 70)
    print("CONFOUNDED CANDIDATES (excluded from main batch, reported separately):")
    for sid, kind, span, note in CONFOUNDED:
        print(f"  [{kind}] {sid:<16} {span!r:<40} -- {note}")


if __name__ == "__main__":
    main()
