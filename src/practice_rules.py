def check_brand_hallucination(extracted_word,start_idx,end_idx,image_context_name):
    word_clean=extracted_word.lower()
    context_clean=image_context_name.lower()
    
    if word_clean not in context_clean:
        return{
            "is_hallucination": True,
            "category":"mischaracterization",
            "span":[start_idx,end_idx],
            "reason": f"Model claimed brand '{extracted_word}', but image context is '{image_context_name}'."
        }
        
    return{"is_hallucination": False}


def check_color_hallucination(extracted_noun,extracted_adj,start_idx,end_idx,visual_ground_truth):
    noun_clean = extracted_noun.lower()
    adj_clean = extracted_adj.lower()

    if noun_clean in visual_ground_truth:
        allowed_color = visual_ground_truth[noun_clean].lower()
        
        if adj_clean != allowed_color:
            return {
                "is_hallucination": True,
                "category": "mischaracterization",
                "span": [start_idx, end_idx],
                "reason": f"Model claimed {noun_clean} is '{extracted_adj}' , but visual it is actually '{allowed_color}'."
            }
    return {"is_hallucination": False}
    
# ==========================================
# TEST CASE 2: The Cauliflower Check
# ==========================================
# This dictionary represents what we KNOW is true about the image
image_knowledge_base = {
    "cauliflower": "purple",
    "truck": "red"
}

# Simulating what Stanza extracted from the text: "the white cauliflower"
target_noun = "cauliflower"
target_adj = "purple"
span_start = 14
span_end = 19

color_result = check_color_hallucination(
    target_noun, target_adj, span_start, span_end, image_knowledge_base
)

print("\n--- COLOR LOGIC TEST RUN ---")
if color_result["is_hallucination"]:
    print(f"🚨 Hallucination Detected!")
    print(f"📍 Target Span: {color_result['span']}")
    print(f"💬 Reason: {color_result['reason']}")
else:
    print("✅ Color description is completely accurate!")

"""    
# test case for brand hallucination            
test_word = "Ford"
start = 51
end = 55
image_file = "red_ford_truck.jpg"

result= check_brand_hallucination(test_word,start,end,image_file)
if result["is_hallucination"]:
    print(f"🚨 Hallucination Detected!")
    print(f"📂 Category: {result['category']}")
    print(f"📍 Location Span: {result['span']}")
    print(f"💬 Reason: {result['reason']}") 
else:
    print("✅ No hallucination detected.")

"""