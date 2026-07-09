import re
import nltk

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand",
}


def normalize_text(text):
    if not text:
        return ""
    return (
        text.replace('\u2019', "'").replace('\u2018', "'")
        .replace('\u201c', '"').replace('\u201d', '"')
    )


def _tokenize_words(text):
    return re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", text)


def isolate_span(span_text, label, response_text):
    """One rule per label type. No stopword-vulnerable pattern matching,
    no context-window scoring."""
    span_text = span_text.strip()
    label = (label or "other").lower()

    if span_text and span_text in response_text:
        words = span_text.split()
        if len(words) == 1 or label not in ("miscounting", "mischaracterization"):
            return span_text

    words = _tokenize_words(span_text)
    if not words:
        return span_text

    if label == "miscounting":
        for w in words:
            if w.isdigit() or w.lower() in NUMBER_WORDS:
                return w
        return span_text  # couldn't confidently isolate a number; don't guess

    if label == "mischaracterization":
        try:
            tagged = nltk.pos_tag(words)
        except Exception:
            tagged = [(w, "UNK") for w in words]
        for w, tag in tagged:
            if tag in ("JJ", "JJR", "JJS"):
                return w
        return span_text  # no adjective found; don't guess

    # invention / OCR / other: the hallucination IS the phrase, don't shrink
    return span_text


def map_spans_to_characters(llm_output, response_text):
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

        idx = normalized_response.find(target)
        if idx == -1:
            idx = normalized_response.lower().find(target.lower())
        if idx == -1:
            continue

        end = idx + len(target)
        overlaps = any(
            not (end <= used_start or idx >= used_end)
            for used_start, used_end in already_used
        )
        if overlaps:
            continue

        already_used.append((idx, end))
        mapped.append({
            "start": idx,
            "end": end,
            "label": label,
            "prob": round(float(prob), 4),
        })

    mapped.sort(key=lambda x: x["start"])
    return mapped