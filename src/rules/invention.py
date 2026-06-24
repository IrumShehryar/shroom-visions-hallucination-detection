def check_invention_hallucination(extracted_nouns,valid_objects_in_image):
    valid_set= {obj.lower() for obj in valid_objects_in_image}
    
    for noun,span in extracted_nouns.items():
        if noun.lower() not in valid_set:
            return {
                "is_hallucination": True,
                "category": "invention",
                "span": span,
                "reason": f"Model invented the object '{noun}', which does not exist in this image."
            }
    return {"is_hallucination": False}

# ==========================================
# TEST CASE 3: Inventing a Cat
# ==========================================
# Ground truth: What is ACTUALLY in the photo
true_objects = ["Truck", "Road", "Tree"]

# Simulated Stanza output: Words mapped to their [start, end] positions
simulated_extracted_nouns = {
    "truck": [36, 41],
    "cat": [46, 49]     # The model claims it sees a cat!
}

invention_result = check_invention_hallucination(simulated_extracted_nouns, true_objects)

print("\n--- INVENTION LOGIC TEST RUN ---")
if invention_result["is_hallucination"]:
    print(f"🚨 Hallucination Detected!")
    print(f"📂 Category: {invention_result['category']}")
    print(f"📍 Location Span: {invention_result['span']}")
    print(f"💬 Reason: {invention_result['reason']}")
else:
    print("✅ All mentioned objects actually exist in the image.")