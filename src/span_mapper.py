def normalize_text(text):
    # Replace smart quotes with straight quotes
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    return text

def find_best_occurrence(span_text, response_text, already_used):
    """
    Finds all occurrences of span_text in response_text.
    Skips positions already used by previous spans.
    Returns the first unused occurrence.
    """
    start = 0
    while True:
        idx = response_text.find(span_text, start)
        if idx == -1:
            return None  # not found at all
        
        end = idx + len(span_text)
        
        # Check if this position overlaps with already used spans
        overlaps = any(
            not (end <= used_start or idx >= used_end)
            for used_start, used_end in already_used
        )
        
        if not overlaps:
            return idx  # found a clean unused position
        
        start = idx + 1  # try next occurrence
    
    return None


def map_spans_to_characters(llm_output, response_text):
    """
    Takes LLM output list and maps span_text to character offsets.
    Handles duplicate span_text occurrences.
    """
    mapped = []
    already_used = []  # tracks (start, end) of already mapped spans
    normalized_response = normalize_text(response_text)
    for item in llm_output:
        span_text = normalize_text(item.get("span_text", "").strip())
        label = item.get("label", "other")
        prob = item.get("prob", 0.5)

        if not span_text:
            continue

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