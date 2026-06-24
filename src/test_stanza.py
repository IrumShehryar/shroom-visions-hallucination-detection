import torch

# --- WORKAROUND FOR PYTORCH 2.6+ SECURITY CHECK ---
# Force torch.load to allow Stanza's internal metadata structures
_orig_load = torch.load
torch.load = lambda *args, **kwargs: _orig_load(*args, **{**kwargs, 'weights_only': False})
# --------------------------------------------------

import stanza

print("Initializing stanza pipeline...")
nlp= stanza.Pipeline(lang='en',processors='tokenize,mwt,pos')

sample_text = "Based on the image, the brand name of the truck is Ford."
doc= nlp(sample_text)

print("\n--- STANZA PIPELINE OUTPUT ---")
for sentence in doc.sentences:
    for word in sentence.words:
        """if word.upos in ['NOUN','PROPN','ADJ']:
            print(f"Word: {word.text:<12} | Type: {word.upos:<5}")"""
        if word.text == "Ford":
            print(f"Found Target word:'{word.text}'")
            print(f"Start Index: {word.start_char}, End Index: {word.end_char}")
            print(f"✂️ Python Slice Verification: '{sample_text[word.start_char:word.end_char]}'")