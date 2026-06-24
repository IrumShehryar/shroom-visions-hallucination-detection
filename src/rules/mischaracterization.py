def check_color_hallucination(extracted_noun, extracted_adj, start_idx, end_idx, visual_ground_truth):
    """
    Verifies if a descriptive adjective (color) matches the ground-truth 
    color for that specific noun (object).
    """
    noun_clean = extracted_noun.lower()
    adj_clean = extracted_adj.lower()
    
    if noun_clean in visual_ground_truth:
        allowed_color = visual_ground_truth[noun_clean].lower()
        if adj_clean != allowed_color:
            return {
                "is_hallucination": True,
                "category": "mischaracterization",
                "span": [start_idx, end_idx],
                "reason": f"Claimed {noun_clean} is '{extracted_adj}', but it is actually '{allowed_color}'."
            }
            
    return {"is_hallucination": False}


def check_brand_hallucination(extracted_word, start_idx, end_idx, image_context_name):
    """
    Checks if an extracted brand name exists inside the image's 
    ground-truth text or file name context.
    """
    word_clean = extracted_word.lower()
    context_clean = image_context_name.lower()
    
    if word_clean not in context_clean:
        return {
            "is_hallucination": True,
            "category": "mischaracterization",
            "span": [start_idx, end_idx],
            "reason": f"Model claimed brand '{extracted_word}', but image context is '{image_context_name}'."
        }
    
    return {"is_hallucination": False}