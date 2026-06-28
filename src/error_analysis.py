import json
import os
from datetime import datetime

def run_error_analysis(
    results_path=r"D:\SHROOM\evaluation_results.json",
    originals_path=r"D:\SHROOM\shroom-visions-data\distrib\shroom-vision.train.en.labeled.jsonl",
    output_path=r"D:\SHROOM\error_analysis_log.txt"
):
    with open(results_path) as f:
        results = json.load(f)

    originals = {}
    with open(originals_path) as f:
        for line in f:
            row = json.loads(line)
            originals[row["id"]] = row

    with open(output_path, "w", encoding="utf-8") as log:
        
        def write(text=""):
            log.write(text + "\n")
            print(text)

        write(f"ERROR ANALYSIS LOG")
        write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        write(f"Total samples: {len(results)}")
        write(f"Zero IoU samples: {sum(1 for r in results if r['iou'] == 0.0)}")
        write(f"Perfect IoU samples: {sum(1 for r in results if r['iou'] == 1.0)}")
        write("="*70)

        # Group by failure type
        false_positives = [r for r in results if r["iou"] == 0.0 and r["pred_count"] > 0 and r["gold_count"] == 0]
        missed = [r for r in results if r["iou"] == 0.0 and r["pred_count"] == 0 and r["gold_count"] > 0]
        wrong_location = [r for r in results if r["iou"] == 0.0 and r["pred_count"] > 0 and r["gold_count"] > 0]

        for section_title, section_cases in [
            ("FALSE POSITIVES (system flagged, gold is clean)", false_positives),
            ("MISSED DETECTIONS (system clean, gold has errors)", missed),
            ("WRONG LOCATION (both have spans but no overlap)", wrong_location)
        ]:
            write(f"\n{'='*70}")
            write(f"{section_title} — {len(section_cases)} cases")
            write("="*70)

            for r in section_cases:
                original = originals.get(r["id"], {})
                response = original.get("response", "")

                write(f"\nID: {r['id']}")
                write(f"PROMPT: {original.get('prompt', '')}")
                write(f"IMAGE: {original.get('image_name', '')}")
                write(f"IoU: {r['iou']}")
                write()
                write("FULL RESPONSE:")
                write(response)
                write()
                write("GOLD LABELS:")
                for label in r["gold_labels"]:
                    span_text = response[label["start"]:label["end"]]
                    write(f"  [{label['start']}:{label['end']}] '{span_text}' | {label['label']} | annotator agreement: {label['prob']:.2f}")
                write()
                write("PREDICTED LABELS:")
                if r["pred_labels"]:
                    for label in r["pred_labels"]:
                        span_text = response[label["start"]:label["end"]]
                        write(f"  [{label['start']}:{label['end']}] '{span_text}' | {label['label']} | confidence: {label['prob']:.2f}")
                else:
                    write("  (none — system returned clean)")
                write()
                write("HAIKU REASONING:")
                write(r.get("haiku_reasoning", "NOT SAVED — re-run to capture"))
                write("-"*70)

    print(f"\nLog saved to: {output_path}")

if __name__ == "__main__":
    run_error_analysis()