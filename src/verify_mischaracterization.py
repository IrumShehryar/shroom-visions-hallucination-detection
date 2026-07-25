"""
Pipeline step: Sonnet-based second-opinion filter for mischaracterization
flags. For each mischaracterization entry Haiku reports (after the numeral
relabel fix has already run), asks Sonnet to confirm/reject the flag
against the image; rejected flags are dropped before span-mapping.

Validated on a 51-sample curated batch (validation_outputs/mischar_verification_full_case_log.md)
and confirmed on a genuinely random 60-sample natural-composition draw
(src/verify_natural_composition.py): despite a lower natural catch rate
(33.3% vs the curated batches' higher rates), the net effect on the real
scorer was positive (Cor +0.086, Cor+Lbl +0.087 on touched rows), because
mischaracterization false positives outnumber true positives roughly 2:1
in the wild.

Costs real money (one Sonnet call per mischaracterization flag) -- OFF by
default everywhere it's wired in. Verdicts are cached by (id, span_text)
so interrupted/resumed runs never re-pay for the same flag twice.
"""
import json
import os

from src.verify_flag_with_sonnet import call_sonnet_verify

VERIFY_CACHE_PATH = r"D:\SHROOM\sonnet_verify_cache.json"


def _load_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache, cache_path):
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _parse_verdict(raw_text):
    clean = raw_text.strip()
    if clean.startswith("```"):
        parts = clean.split("```")
        clean = parts[1] if len(parts) > 1 else clean
        if clean.startswith("json"):
            clean = clean[4:]
    try:
        return json.loads(clean.strip())
    except Exception:
        # Parse failures keep the flag rather than silently drop real
        # signal -- matches how the validation batches treated PARSE_ERROR.
        return {"verdict": "confirm", "confidence": None, "reason": "PARSE_ERROR -- kept flag"}


def verify_label_flags(sample_id, llm_output, image_path, response_text, label, cache_path):
    """Returns a filtered copy of llm_output: entries of the given label that
    Sonnet rejects are dropped, everything else passes through untouched.
    Makes one Sonnet API call per NOT-yet-cached entry of that label."""
    cache = _load_cache(cache_path)
    out = []
    changed = False

    for entry in llm_output:
        if str(entry.get("label", "")).lower() != label:
            out.append(entry)
            continue

        span_text = entry.get("span_text", "")
        cache_key = f"{sample_id}::{span_text}"

        if cache_key in cache:
            verdict_obj = cache[cache_key]
        elif not image_path or not os.path.exists(image_path):
            out.append(entry)
            continue
        else:
            result_text, _, _ = call_sonnet_verify(
                image_path, response_text, span_text,
                entry.get("label", label), entry.get("reason", ""),
            )
            verdict_obj = _parse_verdict(result_text)
            cache[cache_key] = verdict_obj
            changed = True

        if verdict_obj.get("verdict") == "reject":
            continue
        out.append(entry)

    if changed:
        _save_cache(cache, cache_path)
    return out


def verify_mischaracterization_flags(sample_id, llm_output, image_path, response_text, cache_path=VERIFY_CACHE_PATH):
    return verify_label_flags(sample_id, llm_output, image_path, response_text, "mischaracterization", cache_path)
