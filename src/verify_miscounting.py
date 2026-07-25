"""
Pipeline step: Sonnet-based second-opinion filter for miscounting flags.
Same mechanism as src/verify_mischaracterization.py (see that file for the
shared implementation), applied to the miscounting label instead.

Validated on a 28-sample curated dev batch
(validation_outputs/miscounting_verification_devbatch_summary.md): catch rate
25% on known false positives, FN rate 15.4% on known true positives, real
scorer effect +0.0177 Cor / +0.0201 Cor+Lbl on touched rows (true full-row
methodology). Smaller effect than mischaracterization (miscounting's natural
false-positive rate is only ~25%, vs mischaracterization's ~68%), but real
and positive -- unlike invention, which was tested and found net negative
(see attempt_invention_sonnet_verification memory).

Uses a SEPARATE cache file from mischaracterization verification, so the two
can never collide and rerunning one never touches the other's cached spend.
"""
from src.verify_mischaracterization import verify_label_flags

VERIFY_CACHE_PATH = r"D:\SHROOM\sonnet_verify_miscounting_cache.json"


def verify_miscounting_flags(sample_id, llm_output, image_path, response_text, cache_path=VERIFY_CACHE_PATH):
    return verify_label_flags(sample_id, llm_output, image_path, response_text, "miscounting", cache_path)
