def check_invention_hallucination(extracted_nouns, valid_objects_in_image):
    valid_set = {obj.lower() for obj in valid_objects_in_image}

    for noun, span in extracted_nouns.items():
        if noun.lower() not in valid_set:
            return {
                "is_hallucination": True,
                "category": "invention",
                "span": span,
                "reason": f"Model invented the object '{noun}', which does not exist in this image.",
            }

    return {"is_hallucination": False}