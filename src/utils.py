def compress_char_arrays_to_spans(char_probs, char_labels, response_text):
    """
    Converts raw character-level probability and label arrays into aggregated text spans.
    
    Args:
        char_probs (list): Array of floats representing character hallucination probabilities.
        char_labels (list): Array of strings matching the category per character ('Invention', etc.)
        response_text (str): The original generated response string.
        
    Returns:
        list: A list of dicts structured perfectly for the competition jsonl format.
    """
    spans = []
    n = len(char_probs)
    if n == 0:
        return spans
        
    in_span = False
    start_idx = None
    current_label = None
    span_probs = []

    for i in range(n):
        prob = char_probs[i]
        label = char_labels[i]
        
        # We define an active hallucination character if prob > 0.0 and it has a category
        is_hallucination = (prob > 0.0) and (label is not None and label != "")
        
        if is_hallucination:
            if not in_span:
                # Start a brand new span
                in_span = True
                start_idx = i
                current_label = label
                span_probs = [prob]
            elif label == current_label:
                # Continue the current span block
                span_probs.append(prob)
            else:
                # Label changed mid-stream: Close the old span, immediately start the new one
                avg_prob = sum(span_probs) / len(span_probs)
                spans.append({
                    "start": start_idx,
                    "end": i,  # Exclusive boundary
                    "prob": round(avg_prob, 4),
                    "label": current_label.lower().replace(" ", "_") # Matches 'ocr_problem', etc.
                })
                start_idx = i
                current_label = label
                span_probs = [prob]
        else:
            if in_span:
                # Text transitioned back to clean ground-truth: Close active span
                avg_prob = sum(span_probs) / len(span_probs)
                spans.append({
                    "start": start_idx,
                    "end": i,
                    "prob": round(avg_prob, 4),
                    "label": current_label.lower().replace(" ", "_")
                })
                in_span = False
                span_probs = []
                current_label = None

    # Clean up boundary catch if a span ends exactly at the final character
    if in_span:
        avg_prob = sum(span_probs) / len(span_probs)
        spans.append({
            "start": start_idx,
            "end": n,
            "prob": round(avg_prob, 4),
            "label": current_label.lower().replace(" ", "_")
        })

    return spans