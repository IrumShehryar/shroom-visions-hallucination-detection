def check_miscounting_hallucination(claimed_count,true_count,start_idx,end_idx):
    if claimed_count!= true_count:
        return{
            "is_hallucination": True,
            "category":"miscounting",
            "span":[start_idx,end_idx],
            "reason": f"Model claimed there are {claimed_count} items, but there are actually {true_count}."
        }
    return {"is_hallucination": False}

# ==========================================
# TEST CASE 3: Miscounting Objects
# ==========================================
# Let's simulate a scenario where there are 2 boxes in an image,
# but the model text says: "There are three boxes."
true_boxes_count = 2
model_claimed_count = 2  # Extracted from the word "three"
span_start = 10
span_end = 15

count_result = check_miscounting_hallucination(
    model_claimed_count, true_boxes_count, span_start, span_end
)

print("\n--- COUNTING LOGIC TEST RUN ---")
if count_result["is_hallucination"]:
    print(f"🚨 Hallucination Detected!")
    print(f"📂 Category: {count_result['category']}")
    print(f"📍 Location Span: {count_result['span']}")
    print(f"💬 Reason: {count_result['reason']}")
else:
    print("✅ Object counts match up perfectly.")