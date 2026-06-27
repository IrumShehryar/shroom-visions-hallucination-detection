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

