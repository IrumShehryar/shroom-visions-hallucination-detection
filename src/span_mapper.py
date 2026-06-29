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

    span_lower = span_text.lower()
    for num in number_words:
        pattern = r'\b' + num + r'\b'
        match = re.search(pattern, span_lower)
        if match:
            return span_text[match.start():match.end()]

    if label == "mischaracterization" or label=="invention":
        words = span_text.split()
        if len(words) > 3:
            return words[-2:]
        return span_text

    return span_text

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
        
        #span_text = extract_key_span(span_text, label)
        
        
        found_any_match = False

        # 1. Exact case check loop - finds ALL occurrences in the text
        start = 0
        while True:
            idx = normalized_response.find(span_text, start)
            if idx == -1:
                break
            end = idx + len(span_text)
            
            # Verify no structural collisions with already mapped phrases
            overlaps = any(not (end <= used_start or idx >= used_end) for used_start, used_end in already_used)
            
            if not overlaps:
                already_used.append((idx, end))
                mapped.append({
                    "start": idx,
                    "end": end,
                    "label": label,
                    "prob": round(prob, 4)
                })
                found_any_match = True
            
            start = idx + 1  # Move forward by one character to look for duplicate mentions later

        # 2. Case-insensitive fallback loop (only runs if exact case found nothing)
        if not found_any_match:
            start = 0
            span_lower = span_text.lower()
            response_lower = normalized_response.lower()
            
            while True:
                idx = response_lower.find(span_lower, start)
                if idx == -1:
                    break
                end = idx + len(span_text)
                
                overlaps = any(not (end <= used_start or idx >= used_end) for used_start, used_end in already_used)
                
                if not overlaps:
                    already_used.append((idx, end))
                    mapped.append({
                        "start": idx,
                        "end": end,
                        "label": label,
                        "prob": round(prob, 4)
                    })
                    found_any_match = True
                
                start = idx + 1

        if not found_any_match:
            print(f"Warning: '{span_text}' not found in response")

    # Sort final mappings by start index to keep output clean and structured
    mapped.sort(key=lambda x: x["start"])
    return mapped