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
    

def normalize_text(text):
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    return text


def extract_key_span(span_text, label, reason=""):
    span_words = span_text.lower().split()
    reason_text = reason.lower()
    
    span_words_clean = [w.strip(".,*()\"'") for w in span_words if w.strip(".,*()\"'")]
    
    # 1. Explicit contradiction check (Your original logic)
    for word in span_words_clean:
        if f"not {word}" in reason_text or f"instead of {word}" in reason_text or f"no {word}" in reason_text:
            return word

    # 2. Extract using Part-of-Speech tags (Your original logic)
    try:
        tagged = nltk.pos_tag(span_words)
    except:
        tagged = [(w, 'UNK') for w in span_words]

    if label.lower() == "miscounting":
        for word, tag in tagged:
            if tag == 'CD' or word.isdigit():
                return word
                
    elif label.lower() in ["mischaracterization", "invention"]:
        if len(span_words) > 3:
            for word, tag in tagged:
                if tag in ['JJ', 'JJR', 'JJS', 'RB']:
                    return word
            return " ".join(span_words[-2:])
            
    return span_text


def map_spans_to_characters(llm_output, response_text):
    mapped = []
    already_used = []
    normalized_response = normalize_text(response_text)

    for item in llm_output:
        original_span = normalize_text(item.get("span_text", "").strip())
        label = item.get("label", "other")
        prob = item.get("prob", 0.5)
        reasoning = item.get("reason", "")

        if not original_span:
            continue
        
        # Extract the target word using your high-performing logic
        keyword = extract_key_span(original_span, label, reasoning)
        span_tokens = original_span.lower().split()

        # Global search for the extracted keyword
        start_search = 0
        while True:
            idx = normalized_response.lower().find(keyword.lower(), start_search)
            if idx == -1:
                break
            end = idx + len(keyword)

            # Define a local neighborhood context window around the matched keyword
            window_start = max(0, idx - 50)
            window_end = min(len(normalized_response), end + 50)
            context_window = normalized_response.lower()[window_start:window_end]

            # Verify that other helper words from the original span exist in this sentence neighborhood
            matching_context_tokens = sum(1 for t in span_tokens if t != keyword.lower() and t in context_window)

            # If it's a long phrase, it must have at least one context partner nearby to count as a valid match
            is_valid_context = True
            if len(span_tokens) > 1 and matching_context_tokens == 0:
                is_valid_context = False

            if is_valid_context:
                overlaps = any(not (end <= used_start or idx >= used_end) for used_start, used_end in already_used)
                if not overlaps:
                    already_used.append((idx, end))
                    mapped.append({
                        "start": idx,
                        "end": end,
                        "label": label,
                        "prob": round(prob, 4)
                    })
                    break  # Found the true contextual instance, move to next item

            start_search = idx + 1

    mapped.sort(key=lambda x: x["start"])
    return mapped