import re


def normalize_text(text):
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    return text


def extract_key_span(span_text, label):
    number_words = [
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "couple", "several",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    ]

    #if label == "miscounting":
    span_lower = span_text.lower()
    for num in number_words:
        pattern = r'\b' + num + r'\b'
        match = re.search(pattern, span_lower)
        if match:
            return span_text[match.start():match.end()]
        #return span_text

    if label == "mischaracterization":
        words = span_text.split()
        if len(words) > 3:
            return words[-1]
        return span_text

    return span_text


def find_best_occurrence(span_text, response_text, already_used):
    start = 0
    while True:
        idx = response_text.find(span_text, start)
        if idx == -1:
            return None
        end = idx + len(span_text)
        overlaps = any(
            not (end <= used_start or idx >= used_end)
            for used_start, used_end in already_used
        )
        if not overlaps:
            return idx
        start = idx + 1
    return None


def map_spans_to_characters(llm_output, response_text):
    mapped = []
    already_used = []

    normalized_response = normalize_text(response_text)

    for item in llm_output:
        span_text = normalize_text(item.get("span_text", "").strip())
        label = item.get("label", "other")
        prob = item.get("prob", 0.5)

        if not span_text:
            continue

        # Post process to extract shortest meaningful span
        span_text = extract_key_span(span_text, label)

        start = find_best_occurrence(span_text, normalized_response, already_used)

        if start is None:
            print(f"Warning: '{span_text}' not found in response")
            continue

        end = start + len(span_text)
        already_used.append((start, end))

        mapped.append({
            "start": start,
            "end": end,
            "label": label,
            "prob": round(prob, 4)
        })

    return mapped