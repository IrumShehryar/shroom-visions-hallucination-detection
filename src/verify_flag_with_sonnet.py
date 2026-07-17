"""
Targeted verification: instead of a fresh full audit, show Sonnet the image
plus ONE specific flag Haiku already made (span text + Haiku's own reason),
and ask a direct yes/no/confidence verdict on that single claim. Sidesteps
the span-boundary mismatch problem a full independent re-audit would have
(see Appendix D) -- Sonnet judges Haiku's box instead of drawing its own.

Cheap, targeted cost measurement -- run on a handful of known clean-row
false positives to see (a) real token cost of this call shape, and (b)
whether Sonnet's verdict confidence is more separable than Haiku's.
"""
import json
import os
import anthropic
from dotenv import load_dotenv

from src.vision_llm import load_image_as_base64, get_image_media_type, parse_llm_response

load_dotenv("src/.env")

IMAGES_DIR = r"D:\SHROOM\distrib\images\shroom-vis-images"

VERIFY_INSTRUCTIONS = """You are verifying a single flagged claim from a prior review of a vision-language model's response about an image.

RESPONSE (full text the model gave): {response}

A prior review flagged this specific claim as potentially wrong:
  Flagged text: "{span_text}"
  Label: {label}
  Reason given: "{reason}"

Look at the image carefully. Is this specific flag correct -- i.e. is the flagged text actually wrong given what's visible in the image?

Answer with ONLY a JSON object, no other text:
{{"verdict": "confirm" or "reject", "confidence": <0-1>, "reason": "<one sentence>"}}
"confirm" means the flag is correct (the response's claim really is wrong).
"reject" means the flag is incorrect (the response's original claim was actually fine)."""


def call_sonnet_verify(image_path, response_text, span_text, label, reason):
    client = anthropic.Anthropic()
    image_data = load_image_as_base64(image_path)
    media_type = get_image_media_type(image_path)
    prompt = VERIFY_INSTRUCTIONS.format(response=response_text, span_text=span_text, label=label, reason=reason)

    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    raw_text = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            raw_text = block.text
            break
    return raw_text, resp.usage.input_tokens, resp.usage.output_tokens


def main():
    originals = {}
    for line in open(r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl", encoding="utf-8"):
        r = json.loads(line)
        originals[r["id"]] = r

    import glob
    dev_cache = {}
    for path in glob.glob("cache_outputs/cache_*.json"):
        for item in json.load(open(path, encoding="utf-8")):
            dev_cache[item["id"]] = item

    # 4 known clean-row false positives (gold empty, but Haiku flagged something)
    targets = [
        ("train-en-449", "alligator"),
        ("train-en-10800", "white"),
        ("train-en-2037", "not composed of circular tubes"),
        ("train-en-1478", "gold ring"),
    ]

    total_in, total_out = 0, 0
    for sid, target_span in targets:
        row = originals[sid]
        item = dev_cache[sid]
        raw = item.get("haiku_reasoning", "[]").replace("```json", "").replace("```", "").strip()
        try:
            llm_out = json.loads(raw)
        except json.JSONDecodeError:
            llm_out = []
        entry = next((e for e in llm_out if e.get("span_text", "") == target_span), None)
        if entry is None:
            print(f"{sid}: could not find span {target_span!r} in cached llm_out, skipping")
            continue

        image_path = os.path.join(IMAGES_DIR, row.get("image_name", ""))
        print(f"\n{'='*70}\n{sid}  (gold-clean, Haiku flagged {target_span!r} as {entry.get('label')} at {entry.get('prob')})")
        result, in_tok, out_tok = call_sonnet_verify(
            image_path, row["response"], target_span, entry.get("label"), entry.get("reason", "")
        )
        total_in += in_tok
        total_out += out_tok
        print("SONNET VERDICT:", result)
        print(f"  usage: input={in_tok} output={out_tok}")

    print(f"\n{'='*70}")
    n = len(targets)
    print(f"totals: input={total_in} output={total_out} across {n} calls")
    avg_in, avg_out = total_in / n, total_out / n
    cost = avg_in * 2.00 / 1e6 + avg_out * 10.00 / 1e6
    print(f"avg input={avg_in:.0f} avg output={avg_out:.0f}  est. cost/call=${cost:.4f}")


if __name__ == "__main__":
    main()
