# Test-Set Spot-Check Summary

Manual review of predictions from `submission/en.jsonl` (1,201 test samples) against
their actual source images, done after the mischaracterization span-mapper fix was
merged and the final submission regenerated. Samples were drawn from
`test_analysis_outputs/spot_check_sample.txt` (48 samples, 8 per predicted-label
bucket, seed=7) plus a handful of additional samples flagged ad hoc during review.

This is a qualitative check on **test** data (no gold labels available). It is a
separate line of evidence from the quantitative **train**-set IoU/Cor/Cor-lbl
validation (286 + 344 samples, official scorer), which is what the span-mapper fix
was actually validated against before merging.

## Genuine correct catches confirmed against the image

- **`test-en-20347`** (tractor, OCR): response misread hood text; image clearly
  shows "JOHN DEERE". Correct catch.
- **`test-en-20182`** (fondue fork w/ stock-photo watermark, OCR): response wrote
  the photographer's name as "Antan Starikov"; watermark actually reads
  "Anton Starikov". Correct catch.
- Cat with a congenital two-leg condition ("Mercury"): correctly flagged rather
  than accepted as a normal four-legged cat.

## Failure modes identified (real, but individually rare)

1. **Reason/span display-order bug** (log-only, does not affect scoring or the
   submission file). `generate_test_log.py`'s `write_sample_block()` zips a
   position-sorted list (`pred_labels`) against Haiku's original unsorted
   `llm_output` order, so when two flags in one response reorder during
   sorting, their `reason` text gets displayed against the wrong span.
   Confirmed in `test-en-20357` and `test-en-20182`. Not yet fixed (optional,
   cosmetic-only).

2. **Confidence-calibration mismatch**: Haiku assigns high confidence (0.85+)
   to a flag while its own reason admits it can't actually read the disputed
   content. `test-en-9666` (swim goggles OCR flag) — reason says brand text is
   "partially obscured" yet still asserts a specific mismatch with "TYR" at
   0.85 confidence; the same reason also misidentifies the goggles as green
   (green is actually the cap), suggesting the flag is closer to a guess than
   a genuine read.

3. **Typo/generation-error mislabeled as OCR**: Haiku catches its own
   response's grammar/spelling slip (e.g. "This s a...") but tags it as an
   image-reading (OCR) error even though its own reasoning explicitly says
   "this appears to be a transcription or generation error rather than
   reading from the image." Searched across all three cached datasets
   combined (test 1,201 + train-new-prompt 286 + train-old-prompt 344 =
   1,831 samples) for this pattern: 8 total instances, 3 mislabeled as OCR
   (~0.16% of the combined pool). Confirmed real, judged too rare to justify
   a pipeline change.

4. **Miscounting mislabeled as mischaracterization**: Haiku occasionally
   files a wrong-count error under `mischaracterization` instead of
   `miscounting` (e.g. `test-en-20182`'s "only two tines" flag disputes a
   count, not a description). Tested relabeling such flags automatically
   before span isolation as a hypothetical fix — net negative/negligible on
   both training sets (286: +0.00002 IoU, 344: -0.00173 IoU) — not adopted.

5. **Undercounting / miscounting by Haiku itself**: in at least one case
   (`test-en-20182`'s "more than two tines" dispute), Haiku's own count
   claim looks more likely to be wrong than the original response's — the
   image is a fondue fork, conventionally a 2-tine design by definition.

## Overall assessment

The large majority of spot-checked samples are correctly characterized. The
failure modes above are genuine and worth documenting (e.g. for a paper
appendix), but are low-frequency and don't change the conclusion already
established via the official scorer on training data: the span-mapper fix
(dropping POS-tag-based span-shrinking for mischaracterization, keeping
digit-extraction for miscounting) is a net improvement, and the finalized
1,201-sample test submission reflects it correctly.
