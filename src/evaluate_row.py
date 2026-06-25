import re
from src.parse_text import extract_linguistic_features
from src.rules.invention import check_invention_hallucination
from src.rules.mischaracterization import check_color_hallucination, check_brand_hallucination
from src.rules.miscounting import check_miscounting_hallucination       
from src.rules.ocr import check_ocr_hallucination

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
                    char_categories[i]="A"
                    

    # 2. Check Mischaracterization (Category B)
    for adj_item in response_features["adjectives"]:
        if adj_item["lemma"].lower() not in allowed_adjectives:
            for i in range(adj_item["start_idx"], adj_item["end_idx"]):
                if i < response_len:
                    char_probabilities[i] = hallucination_confidence
                    char_categories[i] = "B"

    # 3. Check Miscounting (Category D)
    for num_item in response_features["numerals"]:
        if num_item["lemma"].lower() not in allowed_numerals:
            for i in range(num_item["start_idx"], num_item["end_idx"]):
                if i < response_len:
                    char_probabilities[i] = hallucination_confidence
                    char_categories[i] = "D"

    return {
        "id": dataset_row.get("id"),
        "response": model_response,
        "char_probabilities": char_probabilities,
        "char_categories": char_categories
    }
    
if __name__ == "__main__":
    # Test Case 1: Descriptive filename (ImageNet style). 
    # The model response invents an "airplane" and miscounts the masts.
    # 1. Create a mock row simulating an descriptive image dataset entry
    test_row = {
        "id": "test-001",
        "prompt": "Are there any ships or yachts visible?",
        "image_name": "luxury_yacht_harbor_view.jpg",
        "response": "A red yacht is docked."
    }
    
    print("--- Running Evaluation Test ---")
    output = evaluate_single_row(test_row)
    
    print(f"\nResponse Text: '{output['response']}'")
    print(f"Total Characters: {len(output['response'])}")
    
    print("\nCharacter Breakdown:")
    # Print each character next to its assigned probability and category
    for char_idx, char in enumerate(output["response"]):
        prob = output["char_probabilities"][char_idx]
        cat = output["char_categories"][char_idx]
        # Only highlight if flagged to keep terminal output clean
        flag_status = f"<- Flagged [{cat}] Prob: {prob}" if cat != "O" else ""
        print(f"Index {char_idx:2d} | '{char}' | Cat: {cat} {flag_status}")