"""
Cheap mechanism test (~15 API calls): does a known-missed, late-position span
get caught when it's no longer "late" in a long response?

For each candidate (a real missed invention/mischaracterization gold span,
selected from a long response where the span sits in the last 25-40% of the
text), split the response in half at a sentence boundary near the midpoint,
and audit ONLY the half containing the target span -- using the exact same
prompt_builder + vision_llm call as the real pipeline, same image, same
original question -- as if that half were the entire response.

If position/length decay is the mechanism (not content difficulty), the
catch rate on these previously-100%-missed spans should rise sharply once
they're no longer buried in a long response.
"""
import json
import os
import re

from src.prompt_builder import build_audit_prompt
from src.vision_llm import call_vision_llm
from src.span_mapper import map_spans_to_characters

TRAIN_PATH = r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl"
IMAGES_DIR = r"D:\SHROOM\distrib\images\shroom-vis-images"
CANDIDATES_PATH = r"D:\SHROOM\scratchpad_split_test_candidates.json"
OUT_PATH = r"D:\SHROOM\split_test_results.json"


def get_filename_hint(image_name):
    clean = re.sub(r'[\-_.]', ' ', re.sub(r'\.(jpg|jpeg|png|bmp|gif|webp)$', '', image_name.lower())).strip()
    return f"Background context only — image filename: {clean}. Only use visual evidence." if clean else "No hint"


def split_at_sentence_near_midpoint(response):
    mid = len(response) // 2
    window = response[max(0, mid - 150):mid + 150]
    best_idx = None
    for m in re.finditer(r'[.!?]\s', window):
        idx = max(0, mid - 150) + m.end()
        if best_idx is None or abs(idx - mid) < abs(best_idx - mid):
            best_idx = idx
    split_at = best_idx if best_idx is not None else mid
    return response[:split_at], split_at


def overlaps(a_start, a_end, b_start, b_end):
    return not (a_end <= b_start or a_start >= b_end)


def main():
    originals = {}
    for line in open(TRAIN_PATH, encoding="utf-8"):
        r = json.loads(line)
        originals[r["id"]] = r

    candidates = json.load(open(CANDIDATES_PATH, encoding="utf-8"))
    print(f"Testing {len(candidates)} candidates...")

    results = []
    for i, c in enumerate(candidates, 1):
        sid = c["id"]
        row = originals[sid]
        response = row["response"]
        first_half, split_at = split_at_sentence_near_midpoint(response)
        second_half = response[split_at:]

        target_in_second_half = c["start"] >= split_at
        half_text = second_half if target_in_second_half else first_half
        half_offset = split_at if target_in_second_half else 0

        image_path = os.path.join(IMAGES_DIR, row.get("image_name", ""))
        prompt = build_audit_prompt(row.get("prompt", ""), half_text, get_filename_hint(row.get("image_name", "")))

        print(f"[{i}/{len(candidates)}] {sid} ({c['label']}, target in {'2nd' if target_in_second_half else '1st'} half, half_len={len(half_text)})...")
        try:
            _, raw_response = call_vision_llm(image_path, prompt)
        except Exception as e:
            print(f"  API error: {e}")
            results.append({"id": sid, "error": str(e)})
            continue

        raw = raw_response.replace("```json", "").replace("```", "").strip()
        try:
            llm_out = json.loads(raw)
        except json.JSONDecodeError:
            llm_out = []
        mapped = map_spans_to_characters(llm_out, half_text)

        # rebase to original response coordinates and check overlap with target
        caught = False
        caught_label = None
        for m in mapped:
            m_start = m["start"] + half_offset
            m_end = m["end"] + half_offset
            if overlaps(m_start, m_end, c["start"], c["end"]):
                caught = True
                caught_label = m["label"]
                break

        target_text = response[c["start"]:c["end"]]
        print(f"  target=\"{target_text}\"  caught={caught}" + (f" (as {caught_label})" if caught else ""))

        results.append({
            "id": sid, "gold_label": c["label"], "target_text": target_text,
            "caught": caught, "caught_as_label": caught_label,
            "half_len": len(half_text), "original_resp_len": len(response),
            "original_norm_pos": c["norm_pos"], "raw_response": raw_response,
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n = len([r for r in results if "error" not in r])
    caught_n = sum(1 for r in results if r.get("caught"))
    print(f"\n=== RESULT ===")
    print(f"Previously missed spans re-tested (isolated to their half): {n}")
    print(f"Now caught: {caught_n} ({caught_n/n:.1%})" if n else "n/a")
    print(f"Results written to {OUT_PATH}")


if __name__ == "__main__":
    main()
