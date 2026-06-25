import stanza
import torch
from functools import partial

# Force torch.load to default to weights_only=False for older model formats
torch.load = partial(torch.load, weights_only=False)

nlp= stanza.Pipeline(lang='en',processors='tokenize,lemma,pos',verbose=False)

def extract_linguistic_features(text):
    
    doc=nlp(text)

    extracted_data={
        "nouns":[],
        "adjectives":[],
        "numerals":[]
    }
    
    for sentence in doc.sentences:
        for word in sentence.words:
            
            word_info={
                "text": word.text,
                "lemma": word.lemma,
                "start_idx":word.start_char,
                "end_idx": word.end_char,
            }
            
            if word.upos == "NOUN" or word.upos == "PROPN":
                extracted_data["nouns"].append(word_info)
            elif word.upos == "ADJ":
                extracted_data["adjectives"].append(word_info)  
            elif word.upos == "NUM":
                extracted_data["numerals"].append(word_info)    
    return extracted_data

if __name__ == "__main__":
    sample_text = "Based on the image, the ship does not have traditional masts like those found on sailing vessels. \n\n"
    "However, it does have a communications mast – a relatively small, single pole structure visible towards the top of the ship. "
    "This mast is used for antennas and communication equipment, not for sails. It's a common feature on modern motor yachts like this one.\n\n"
    "So, while it doesn't have masts in the traditional sense, it does have a mast-like structure for technical purposes."
    
    print(f"Testing Stanza parsing on: '{sample_text}'\n")
    
    features = extract_linguistic_features(sample_text)
    
    import json
    print(json.dumps(features, indent=2))
            