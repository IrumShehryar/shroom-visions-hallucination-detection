# SHROOM-Vis Baseline

This repository contains the Hallushift++ baseline for the SHROOM-Vis hallucination detection task. The baseline extracts token-level signals from a vision-language model response, aligns those tokens with hallucination labels, trains token-level classifiers, and converts the predictions into the JSONL format required by the official evaluation script.

The main entry point is the notebook:

```text
baseline.ipynb
```

## What the baseline does

The baseline follows these steps:

1. Loads the labeled SHROOM-Vis data from `data/`.
2. Parses span-level hallucination labels.
3. Builds intermediate JSONL source files under `notebook_jsonl_sources/`.
4. Indexes the image folder `shroom-vis-images/` and matches every example to its image.
5. Loads the LLaVA model through `scripts/llava.py`.
6. Extracts token-level statistics from each model response, including token probabilities, negative log-likelihood, perplexity, hidden-state norms, and attention summaries.
7. Aligns token-level features with gold hallucination spans.
8. Saves per-token feature files under `hidden/`.
9. Runs a grid search over several classifiers.
10. Saves the grid-search results to CSV.
11. Converts token-level predictions into the final JSONL submission format.

## Required data

Place the SHROOM-Vis image folder in the repository root:

```text
shroom-vis-images/
```

The notebook searches this folder recursively and matches images by filename. The image folder must contain the actual image files referenced by the dataset rows.

The labeled dataset files should be placed in:

```text
data/
```

The notebook reads every file in `data/` whose filename ends with:

```text
labeled.jsonl
```

Each row is expected to contain at least the following fields:

```text
id
split
language
image_name
prompt
response
labels
```

For final submission conversion, the notebook also expects reference hidden test files under:

```text
data/hidden/
```

## Notes

- The notebook currently uses LLaVA as the model backend.
- Token-to-character alignment is approximate and depends on tokenizer behavior.
- The full feature extraction step is expensive; use `SMOKE_TEST` or `max_rows` for debugging.
