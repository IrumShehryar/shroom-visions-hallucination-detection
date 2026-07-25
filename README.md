# SHROOM-Visions Hallucination Detection Pipeline

This repo implements the English-language detection pipeline for **[SHROOM-Visions 2026](https://helsinki-nlp.github.io/shroom/2026)** — the "Shared-task on Hallucinations and Related Observable Overgeneration Mistakes in vision language models," run by Helsinki NLP.

## What the task asks for

Given an image, a text prompt, and a vision-language model's generated response to that prompt, participants must find every **hallucinated span** in the response — text that isn't supported by the image — and for each one report:

- the character offsets of the span,
- which of five categories it falls into:
  - **Invention** — an entity, object, property, or event that isn't in the image at all
  - **Mischaracterization** — something in the image, but described incorrectly
  - **OCR** — text visible in the image, misread in the response
  - **Miscounting** — a wrong quantity for something visible
  - **Other** — a hallucination that doesn't fit the above
- a confidence/probability score for that flag.

Submissions are scored on two character-level metrics: span-identification via IoU, and calibration via Spearman correlation between predicted probabilities and empirical multi-annotator agreement. The organizers' full dataset spans four languages (Chinese, English, French, Italian) with outputs from five different vision-language models; this repo targets the English subset.

## Pipeline

```
image + prompt + response
        |
        v
  Haiku 4.5 (src/vision_llm.py, src/prompt_builder.py)
  zero-shot audit -> flagged spans + category + confidence
        |
        v
  Numeral relabeling (src/relabel_miscounting.py)
  short mischaracterization spans containing a numeral -> miscounting
        |
        v
  Sonnet 5 second-opinion verification, mischaracterization/miscounting only
  (src/verify_mischaracterization.py, src/verify_miscounting.py,
   src/verify_flag_with_sonnet.py) -> confirm/reject each flag
        |
        v
  Span mapping (src/span_mapper.py)
  maps reported span text back to character offsets in the response
        |
        v
  submission JSONL (src/generate_test_predictions.py, src/format_checker.py)
```

Haiku 4.5 does the initial zero-shot detection pass over every sample. Two cheap deterministic/LLM-assisted fixes are layered on top before spans are mapped to character offsets and validated against the official submission format: a numeral-based relabeling rule that catches a systematic mischaracterization/miscounting confusion, and a Sonnet 5 verification step that re-checks Haiku's mischaracterization and miscounting flags against the image before they're kept.

Both post-processing steps were validated on held-out labeled data before being deployed, and both produced a confirmed real gain on the leaderboard (Cor+Lbl 0.2868 → 0.3015, Cor 0.3816 → 0.3949 on the final English test-set submission).

## Repo scope

This repo tracks only the live pipeline files that produce a submission — not the research/validation trail (dev-set error analysis, ablations, chart generation, manual review tooling) used to arrive at it.

## Setup

Requires `anthropic`, `python-dotenv`, and `nltk` (with `averaged_perceptron_tagger_eng` and `punkt_tab` downloaded). Set `ANTHROPIC_API_KEY` in a `.env` file at the repo root. The organizers' dataset and images aren't included here — point `TEST_DATA_PATH` and `IMAGES_DIR` in `src/generate_test_predictions.py` at your local copy.
