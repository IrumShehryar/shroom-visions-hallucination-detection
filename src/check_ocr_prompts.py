import json

count = 0
with open(r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl") as f:
    for line in f:
        row = json.loads(line)
        has_ocr = any(l["label"].lower() in ["ocr","ocr_problem"] for l in row.get("labels",[]))
        if has_ocr:
            print(row["prompt"])
            count += 1
        if count >= 20:
            break