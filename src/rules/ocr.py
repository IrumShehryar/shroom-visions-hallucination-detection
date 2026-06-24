def check_ocr_hallucination(extracted_text,true_sign_text,start_idx,end_idx):
    model_read=extracted_text.strip().lower()
    actual_text= true_sign_text.strip().lower()
    
    if model_read != actual_text:
        return{
            "is_hallucination": True,
            "category":"misreading",
            "span":[start_idx,end_idx],
            "reason": f"Model read text as '{extracted_text}', but the image actually says '{true_sign_text}'."
        }
    return {"is_hallucination": False}

# ==========================================
# TEST CASE 4: Misreading a Sign
# ==========================================
# Imagine a stop sign in the image, but the model reads it as "SHOP"
ground_truth_sign = "STOP"
model_extracted_text = "STOP" 
span_start = 20
span_end = 24

ocr_result = check_ocr_hallucination(
    model_extracted_text, ground_truth_sign, span_start, span_end
)

print("\n--- OCR LOGIC TEST RUN ---")
if ocr_result["is_hallucination"]:
    print(f"🚨 Hallucination Detected!")
    print(f"📂 Category: {ocr_result['category']}")
    print(f"📍 Location Span: {ocr_result['span']}")
    print(f"💬 Reason: {ocr_result['reason']}")
else:
    print("✅ Text elements match perfectly.")