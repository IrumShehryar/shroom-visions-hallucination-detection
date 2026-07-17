"""
Writes a SMALL, curated review file instead of the full 1201-sample log --
picks N random samples per predicted-label bucket so you can read top to
bottom without searching. Same per-sample format as generate_test_log.py.

Run this AFTER generate_test_log.py (or generate_test_predictions.py directly --
it rebuilds records itself, it just doesn't need the full log to exist first).
"""

import os
import random

from src.generate_test_log import build_records, group_by_bucket, write_sample_block, OUTPUT_DIR

RANDOM_SEED = 7
PER_BUCKET = 8


def main():
    random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    records = build_records()
    by_bucket = group_by_bucket(records)

    out_path = os.path.join(OUTPUT_DIR, "spot_check_sample.txt")
    total = 0
    with open(out_path, "w", encoding="utf-8") as log:
        log.write("TEST SET SPOT-CHECK SAMPLE (random subset, no gold labels)\n")
        log.write(f"Up to {PER_BUCKET} random samples per predicted-label bucket, seed={RANDOM_SEED}\n")
        log.write("=" * 70 + "\n")

        for bucket, items in sorted(by_bucket.items(), key=lambda kv: -len(kv[1])):
            pool = list(items)
            random.shuffle(pool)
            picks = pool[:PER_BUCKET]
            total += len(picks)
            log.write(f"\n\n{'#' * 70}\nBUCKET: {bucket.upper()} ({len(picks)} of {len(items)} total)\n{'#' * 70}\n")
            for r in picks:
                write_sample_block(log, r)

    print(f"Wrote {total} spot-check samples to: {out_path}")


if __name__ == "__main__":
    main()
