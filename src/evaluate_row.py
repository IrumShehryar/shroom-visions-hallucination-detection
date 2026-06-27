import re
import os
from src.parse_text import extract_linguistic_features
from src.prompt_builder import build_audit_prompt
from src.vision_llm import call_vision_llm
from src.span_mapper import map_spans_to_characters

NUMERIC_MAPPING = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
}

def clean_and_check_filename(filename):
    filename_lower = filename.lower()
    filename_clean = re.sub(r'\.(jpg|jpeg|png|bmp|gif|webp)$', '', filename_lower)
    filename_clean = re.sub(r'[\-_.]', ' ', filename_clean).strip()
    words = filename_clean.split()
    if not words:
        return filename_clean, False
    alpha_words = [w for w in words if w.isalpha() and len(w) > 1]
    is_descriptive = (len(alpha_words) / len(words)) >= 0.2
    return filename_clean, is_descriptive

def get_filename_hint(image_name):
    clean_filename, is_descriptive = clean_and_check_filename(image_name)
    if is_descriptive:
        return f"The image filename suggests it contains: {clean_filename}"
    return "No descriptive filename available"

def get_image_path(image_name, images_dir=r"D:\SHROOM\distrib\images\shroom-vis-images"):
    return os.path.join(images_dir, image_name)

def evaluate_single_row(dataset_row):
    prompt_text = dataset_row.get("prompt", "")
    image_name = dataset_row.get("image_name", "")
    model_response = dataset_row.get("response", "")

    # Step 1: Get filename hint
    filename_hint = get_filename_hint(image_name)

    # Step 2: Build audit prompt
    audit_prompt = build_audit_prompt(
        prompt_text=prompt_text,
        response_text=model_response,
        filename_hint=filename_hint
    )

    # Step 3: Get image path
    image_path = get_image_path(image_name)

    # Step 4: Call vision LLM
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return {"id": dataset_row.get("id"), "labels": []}

    llm_output = call_vision_llm(image_path, audit_prompt)

    # Step 5: Map spans to character offsets
    labels = map_spans_to_characters(llm_output, model_response)

    return {
        "id": dataset_row.get("id"),
        "labels": labels
    }


"""import re
from src.parse_text import extract_linguistic_features
from src.rules.invention import check_invention_hallucination
from src.rules.mischaracterization import check_color_hallucination, check_brand_hallucination
from src.rules.miscounting import check_miscounting_hallucination       
from src.rules.ocr import check_ocr_hallucination
from src.utils import compress_char_arrays_to_spans
# Global numeric mapping to align string numbers with digits
NUMERIC_MAPPING = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
}

def clean_and_check_filename(filename):
    
    filename_lower=filename.lower()
    filename__clean=re.sub(r'\.(jpg|jpeg|png|bmp|gif|webp)$', '', filename_lower)
    filename_clean=re.sub(r'[\-_.]', ' ', filename__clean).strip()
    
    words=filename_clean.split()
    
    if not words: 
        return filename_clean, False
    
    alpha_words=[w for w in words if w.isalpha() and len(w)>1]
    is_descriptive= (len(alpha_words) / len(words))  >= 0.2
    return filename_clean, is_descriptive

def evaluate_single_row(dataset_row):
    prompt_text=dataset_row.get("prompt","")
    image_name=dataset_row.get("image_name","")
    model_response = dataset_row.get("response","")
    
    allowed_nouns=set()
    allowed_adjectives=set()
    allowed_numerals=set()  
    
    # Step 1: Extract features from the Prompt
    if prompt_text:
        p_features = extract_linguistic_features(prompt_text)
        allowed_nouns.update([w["lemma"].lower() for w in p_features["nouns"]])
        allowed_adjectives.update([w["lemma"].lower() for w in p_features["adjectives"]])
        allowed_numerals.update([w["lemma"].lower() for w in p_features["numerals"]])
    
    # Step 2: Clean filename once and determine its type
    clean_filename,is_descriptive=clean_and_check_filename(image_name)
    
    # Step 3: Add filename descriptors to allowed sets if valid
    if is_descriptive:
        f_features= extract_linguistic_features(clean_filename)
        allowed_nouns.update([w["lemma"].lower() for w in f_features["nouns"]])
        allowed_adjectives.update([w["lemma"].lower() for w in f_features["adjectives"]])
        allowed_numerals.update([w["lemma"].lower() for w in f_features["numerals"]])   
       
    # Step 4: Parse the Model Response to check for matches 
    response_features= extract_linguistic_features(model_response)
    
    # Initialize both character-level arrays matching the response length
    response_len=len(model_response)
    char_probabilities=[0.0]*response_len
    char_categories=["O"]* response_len
    
    hallucination_confidence=0.90 if is_descriptive else 0.40
    
    # Check Invention( Category A)
    for noun_item in response_features["nouns"]:
        if noun_item["lemma"].lower() not in allowed_nouns:
            for i in range(noun_item["start_idx"],noun_item["end_idx"]):
                if i< response_len:
                    char_probabilities[i]=hallucination_confidence
                    char_categories[i]="Invention"
                    

    # 2. Check Mischaracterization (Category B)
    for adj_item in response_features["adjectives"]:
        if adj_item["lemma"].lower() not in allowed_adjectives:
            for i in range(adj_item["start_idx"], adj_item["end_idx"]):
                if i < response_len:
                    char_probabilities[i] = hallucination_confidence
                    char_categories[i] = "Mischaracterization"

    # 3. Check Miscounting (Category D)
    for num_item in response_features["numerals"]:
        num_lemma = num_item["lemma"].lower()
        # Convert word form to digit form if applicable (e.g., "three" -> "3")
        normalized_num = NUMERIC_MAPPING.get(num_lemma, num_lemma)
        normalized_allowed = {NUMERIC_MAPPING.get(n, n) for n in allowed_numerals}
        
        if normalized_num not in normalized_allowed:
            for i in range(num_item["start_idx"], num_item["end_idx"]):
                if i < response_len:
                    char_probabilities[i] = hallucination_confidence
                    char_categories[i] = "Miscounting"
    # =============================================================
    # 4. RUN COMPRESSION UTILITY & FORMAT FOR OFFICIAL SUBMISSION
    # =============================================================
    # Converts arrays into: [{"start": X, "end": Y, "prob": Z, "label": "..."}]
    compressed_labels = compress_char_arrays_to_spans(char_probabilities, char_categories, model_response)
    return {
        "id": dataset_row.get("id"),
        "labels":compressed_labels,
    }
    
#if __name__ == "__main__":
    # Test Case 1: Descriptive filename (ImageNet style). 
    # The model response invents an "airplane" and miscounts the masts.
    # 1. Create a mock row simulating an descriptive image dataset entry
"""