# SHROOM-Visions 2026: When One Fix Doesn't Fit All

**Category-Aware Hallucination Span Detection in Vision-Language Models**

This repo implements my submission to **[SHROOM-Visions 2026](https://helsinki-nlp.github.io/shroom/2026)** — the "Shared-task on Hallucinations and Related Observable Overgeneration Mistakes in vision language models," run by Helsinki NLP — targeting the English subset. It accompanies a paper of the same title describing the system, the empirical validation methodology behind each design decision, and a set of negative results (rejected fixes, and *why* they were rejected) that turned out to be as informative as the ones that worked.

**Final result: Cor+Lbl 0.3015, Cor 0.3949, IoU 0.3421 — ranked 13th of 28 (Cor+Lbl) / 15th of 28 (Cor).**

## What the task asks for

Given an image, a text prompt, and a vision-language model's generated response to that prompt, the goal is to find every **hallucinated span** in the response — text that isn't supported by the image — and for each one report:

- the character offsets of the span,
- which of five categories it falls into:
  - **Invention** — an entity, object, property, or event that isn't in the image at all
  - **Mischaracterization** — something in the image, but described incorrectly
  - **OCR** — text visible in the image, misread in the response
  - **Miscounting** — a wrong quantity for something visible
  - **Other** — a hallucination that doesn't fit the above
- a confidence/probability score for that flag.

Submissions are ranked by three character-level metrics: **Cor+Lbl** (Spearman correlation between predicted and gold hallucination probabilities, requiring the predicted category to also match — the primary ranking metric), **Cor** (the same measure without the category requirement), and **IoU** (character-wise span overlap). The organizers' dataset (SHEEP; Mickus et al., 2026) spans four languages (Chinese, English, French, Italian) with outputs from five different vision-language models plus human-written hallucination samples; this repo targets the English subset (3,799 labeled train rows, 1,201 unlabeled test rows).

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

Both post-processing steps were validated on held-out labeled data before being deployed, not just on the data they were designed against — a distinction that mattered, since three other candidate filters looked like real improvements on development data and reversed sign on a genuine held-out set (see Research Highlights below).

## Results

| Stage | Cor+Lbl | Cor | IoU |
|---|---|---|---|
| Baseline (Haiku 4.5, zero-shot) | 0.2868 | 0.3816 | 0.3287 |
| + Numeral relabeling fix | 0.2897 | 0.3824 | 0.3284 |
| + Mischaracterization verification | 0.3008 | 0.3934 | 0.3384 |
| + Miscounting verification | **0.3015** | **0.3949** | **0.3421** |

Final leaderboard rank: **13th of 28 teams (Cor+Lbl)**, 15th of 28 (Cor).

## Research highlights

Beyond the pipeline itself, the project's paper documents an empirical validation methodology applied to every candidate fix, and reports negative results with the same rigor as positive ones:

- **A dev-vs-held-out discipline that caught real overfitting.** Three post-processing filters (a confidence threshold, hedge-language suppression, negation-sentence suppression) each looked like a genuine improvement on development data (+1.8% to +11.0% Cor+Lbl) and **reversed sign on held-out data** (−2.5% to −13.7%) — all three were correctly rejected because the pipeline was never trusted to a single validation split.
- **A controlled mechanism test, not just an aggregate metric, to diagnose *why* detections were missed.** Re-auditing 15 known-missed gold spans with only half the response shown (as if it were the whole response) found the model still flagged the right sentence 100% of the time but matched gold's exact span boundaries only 13.3% of the time — ruling out an attention-decay explanation in favor of a genuine span-boundary granularity mismatch.
- **Two independently-validated fixes that survived the held-out check**, deployed to the real submission: a syntactic numeral-based relabeling rule, and a second-opinion LLM verification step (Claude Sonnet 5 re-checking Claude Haiku 4.5's own flags) — the latter generalized to mischaracterization and miscounting but was tested and *rejected* for invention after it came back net-negative on the real scorer (Cor −0.0143, Cor+Lbl −0.0230), rather than assumed to generalize.
- **A cost-aware experimental decision, not just a performance one.** An independent second-model recall pass (Sonnet auditing from scratch, merged via union) showed a real positive effect on a natural sample (+0.0215 Cor+Lbl), but at an estimated $16.70 for full deployment, exceeding the remaining budget — reported as a validated-but-undeployed finding rather than silently dropped.
- **A structural limitation surfaced through direct experiment, not assumption.** Four different VLMs (Gemini, GPT-5.5 Instant, Claude Haiku 4.5, Claude Sonnet 5) were separately asked to count flags in the same image, outside the evaluation pipeline entirely — all four miscounted, despite the dataset's own response and gold label being correct, demonstrating that visual counting is unreliable across models generally, not an artifact of one detector.

## Repo scope

This repo tracks only the live pipeline files that produce a submission — not the research/validation trail (dev-set error analysis, ablations, chart generation, manual review tooling) used to arrive at it.

## Setup

Requires `anthropic`, `python-dotenv`, and `nltk` (with `averaged_perceptron_tagger_eng` and `punkt_tab` downloaded). Set `ANTHROPIC_API_KEY` in a `.env` file at the repo root. The organizers' dataset and images aren't included here — point `TEST_DATA_PATH` and `IMAGES_DIR` in `src/generate_test_predictions.py` at your local copy.

## How to run

From the repo root:

```
python -m src.generate_test_predictions
```

This calls Haiku 4.5 once per sample in `TEST_DATA_PATH`, checkpointing raw responses to a local cache every 25 samples so a crash or interruption doesn't cost you re-calls — rerunning the command resumes from the cache instead of re-querying already-processed samples. Before spending any API calls, it prints how many samples remain and asks for `y`/`n` confirmation.

Once all samples are cached, it applies the numeral-relabeling fix, runs Sonnet 5 verification on mischaracterization/miscounting flags (toggle with `ENABLE_SONNET_VERIFICATION` / `ENABLE_MISCOUNTING_VERIFICATION` at the top of the file — each has its own cache, so disabling one doesn't affect the other's cached verdicts), maps spans to character offsets, writes the submission JSONL to `SUBMISSION_OUT`, and validates it with `format_checker.py` before reporting it ready to submit.
