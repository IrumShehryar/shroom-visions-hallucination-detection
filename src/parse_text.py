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

