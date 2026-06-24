import json
file_path= "shroom-visions-data/distrib/shroom-vision.train.en.labeled.jsonl"

total_count=0
hallucination_count=0

with open(file_path, 'r',encoding='utf-8') as f:
    for line in f:
        sample = json.loads(line)
        total_count+=1
        
        if sample.get('labels'):
            hallucination_count+=1
clean_count=total_count-hallucination_count
print("--- DATASET BALANCE REPORT ---")
print(f"Total Samples analyzed: {total_count}")
print(f"Clean Responses:        {clean_count} ({(clean_count/total_count)*100:.1f}%)")
print(f"Hallucinated Responses: {hallucination_count} ({(hallucination_count/total_count)*100:.1f}%)")