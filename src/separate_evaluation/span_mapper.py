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
    
    # Clean punctuation from words for clean matching
    span_words_clean = [w.strip(".,*()\"'") for w in span_words if w.strip(".,*()\"'")]
    
    # 1. Check if the reasoning explicitly points to a replacement phrase
    for word in span_words_clean:
        if f"not {word}" in reason_text or f"instead of {word}" in reason_text or f"no {word}" in reason_text:
            return word

    # 2. Extract specific targets using Part-of-Speech tags
    try:
        tagged = nltk.pos_tag(span_words)
    except:
        tagged = [(w, 'UNK') for w in span_words] # Safe fallback if nltk data isn't loaded

    if label.lower() == "miscounting":
        # Pull the absolute number (Cardinal Number)
        for word, tag in tagged:
            if tag == 'CD' or word.isdigit():
                return word
                
    elif label.lower() in ["mischaracterization", "invention"]:
        # Fallback for short descriptions or pull the dominant descriptive adjective/adverb
        if len(span_words) > 3:
            # Check for adjectives
            for word, tag in tagged:
                if tag in ['JJ', 'JJR', 'JJS', 'RB']:
                    return word
            return " ".join(span_words[-2:]) # Fallback to last two words if no explicit tag hit
            
    return span_text # Safety fallback to original span

def map_spans_to_characters(llm_output, response_text):
    mapped = []
    already_used = []

    normalized_response = normalize_text(response_text)

    for item in llm_output:
        span_text = normalize_text(item.get("span_text", "").strip())
        label = item.get("label", "other")
        prob = item.get("prob", 0.5)
        reasoning = item.get("reason", "")

        if not span_text:
            continue
        
        span_text = extract_key_span(span_text, label,reasoning)
        
        
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