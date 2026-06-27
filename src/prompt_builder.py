def build_audit_prompt(prompt_text,response_text,filename_hint):
    role="""You are a hallucination detection expert analyzing outputs from vision-language models.

        You will be given:
        - An image
        - A prompt that was asked about the image
        - A response that a vision-language model generated

        Your job is to identify which parts of the response are 
        factually wrong or unsupported by the image."""
    
    categories="""There are five hallucination categories:
        INVENTION: The response mentions something that does not 
        exist in the image at all. This includes invented locations, 
        objects, or facts that cannot be verified from the image.
        Example: Claiming the image shows Auckland New Zealand 
        when no location markers are visible.

        MISCHARACTERIZATION: Something real in the image is 
        described incorrectly. The object exists but a property 
        is wrong — color, material, direction, name.
        Example: Saying cables are attached to the ceiling when 
        they are attached to the wall.

        MISCOUNTING: The response states a wrong quantity of 
        something visible in the image.
        Example: Saying "four legs" when the animal has three.

        OCR_PROBLEM: The response misreads text that is visibly 
        written in the image.
        Example: Reading a slide as "sign on" when it says "log in".
        When flagging spans follow these rules:

        OTHERS: Any other hallucination that does not fit the above categories.
    """
    span_rules=span_rules="""SPAN EXTRACTION RULES:

        RULE 1 - MISCOUNTING: flag ONLY the number word itself.
        Example: "four legs" → span_text = "four"
        Example: "two front legs" → span_text = "two"
        Find ALL wrong number words, each as a separate entry.

        RULE 2 - MISCHARACTERIZATION: flag ONLY the wrong adjective or property word.
        Example: "cables attached to the ceiling" → span_text = "ceiling"

        RULE 3 - INVENTION: flag ONLY the invented name or place.
        Example: "this is Auckland New Zealand" → span_text = "Auckland New Zealand"

        RULE 4 - OCR_PROBLEM: flag ONLY the misread word.

        RULE 5 - Do NOT flag background knowledge or explanations.

        CRITICAL: span_text must be the shortest possible substring. 
        Never return a full sentence.
        """

        
    ocr_guide="""If the prompt asks about text visible in the image:
        - Read the text in the image very carefully before 
        evaluating the response
        - Compare word by word against what the response claims
        - Flag any word that differs from what the image shows
        -If the image text is too blurry to read confidently, 
        do not flag OCR errors — assign low confidence only
        """
    
    confidence_guide="""For each flagged span assign a confidence score between 0 and 1:

        0.8 - 1.0: You are certain this is wrong. The image clearly 
        contradicts this claim. Any reasonable person 
        looking at the image would agree.

        0.5 - 0.7: You are fairly confident this is wrong but there 
        is some ambiguity. The image suggests it is wrong 
        but interpretation is needed.

        0.2 - 0.4: You think this might be wrong but you are not sure. 
        The claim is questionable but the image does not 
        clearly contradict it.

        If you are genuinely unsure whether something is wrong, 
        assign low confidence rather than not flagging it.
        If nothing is clearly wrong, return an empty list.
        Be conservative — do not flag things unless you have 
        clear visual evidence they are wrong.

    """
   
    output_format="""Return your answer as a JSON array only. No other text.
        Each entry must have:
        - "span_text": the exact text from the response that is wrong
        - "label": one of invention, mischaracterization, 
            miscounting, ocr_problem, other
        - "prob": your confidence score between 0 and 1
        - "reason": one sentence explaining why it is wrong

        If nothing is hallucinated return exactly: []
    """
    input_section= f"""
    FILENAME HINT: {filename_hint}
    PROMPT: {prompt_text}
    RESPONSE: {response_text}
    """
    full_prompt = f"{role}\n\n{categories}\n\n{span_rules}\n\n{ocr_guide}\n\n{confidence_guide}\n\n{output_format}\n\n{input_section}"
    return full_prompt