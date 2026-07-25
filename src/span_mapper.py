import re
import string
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


BOUNDARY_PUNCT = string.whitespace + ".,:;/\\'\"()[]{}<>|"


def isolate_span(span_text, label, response_text):
    """One rule per label type. No stopword-vulnerable pattern matching,
    no context-window scoring."""
    # Haiku sometimes wraps a span in incidental separator punctuation
    # (leading "." before "com", leading "://" on a URL fragment) that
    # isn't part of the actual hallucinated content -- trim it so the
    # mapped offsets align with how spans are annotated.
    stripped = span_text.strip(BOUNDARY_PUNCT)
    span_text = stripped if stripped else span_text.strip()
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

    # mischaracterization / invention / OCR / other: the hallucination IS
    # the phrase Haiku reported -- don't try to algorithmically guess which
    # word within it is "the" wrong one. Tested against labeled data:
    # POS-tagging for an adjective and returning just that word measurably
    # regresses IoU relative to trusting the reported span verbatim.
    return span_text


def _find_unused_occurrence(haystack, needle, already_used):
    """Like str.find, but skips past occurrences that overlap an already-claimed range,
    so repeated mentions of the same short span_text (e.g. "four" appearing 3 times)
    map to their own distinct occurrences instead of all colliding on the first one."""
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            return -1
        end = idx + len(needle)
        overlaps = any(not (end <= used_start or idx >= used_end) for used_start, used_end in already_used)
        if not overlaps:
            return idx
        start = idx + 1


def map_spans_to_characters(llm_output, response_text, sample_id=None, image_path=None, verify_mischar=False, verify_miscounting=False):
    # Local import: relabel_miscounting imports NUMBER_WORDS/_tokenize_words
    # from this module, so a top-level import here would be circular.
    from src.relabel_miscounting import relabel_miscounting
    llm_output = relabel_miscounting(llm_output)

    # Optional Sonnet second-opinion filters -- cost real API calls, so they
    # only run when explicitly enabled with an image_path. Each uses its own
    # cache file (see src/verify_mischaracterization.py, src/verify_miscounting.py)
    # so enabling one never re-pays for or touches the other's cached spend.
    if verify_mischar and image_path:
        from src.verify_mischaracterization import verify_mischaracterization_flags
        llm_output = verify_mischaracterization_flags(sample_id, llm_output, image_path, response_text)
    if verify_miscounting and image_path:
        from src.verify_miscounting import verify_miscounting_flags
        llm_output = verify_miscounting_flags(sample_id, llm_output, image_path, response_text)

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
        mapped.append({
            "start": idx,
            "end": end,
            "label": label,
            "prob": round(float(prob), 4),
        })

    mapped.sort(key=lambda x: x["start"])
    return mapped