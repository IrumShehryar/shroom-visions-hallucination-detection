import json
file_path= "shroom-visions-data/distrib/shroom-vision.train.en.labeled.jsonl"

def peek_at_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Read just the first line (row 0)
            first_line = f.readline()
            
            # Parse the string into a clean Python dictionary
            data_row = json.loads(first_line)
            
            print("Successfully opened the file! Here is the first row:\n")
            print(json.dumps(data_row, indent=2))
            
    except FileNotFoundError:
        print(f"Error: Could not find the file at {file_path}. Please check the file name.")

if __name__ == "__main__":
    peek_at_data(file_path)
"""total_count=0
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
"""