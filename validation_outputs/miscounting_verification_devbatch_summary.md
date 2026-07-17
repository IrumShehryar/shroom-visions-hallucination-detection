# Miscounting-Label Sonnet Verification — Bounded Dev-Batch Check

Generated from `verify_miscounting_devbatch_results.json` (28-sample curated
batch: 13 known FP + 15 known TP, seed=9090). Same bounded-check discipline
used for the invention test.

## Natural population (free, no API calls)

| | count | % |
|---|---|---|
| Total predicted miscounting spans (551-row labeled pool) | 116 | 100% |
| TP (overlaps gold miscounting span) | 87 | 75.0% |
| FP (no gold overlap) | 29 | 25.0% |

Much cleaner base rate than mischaracterization (68% FP) or invention (75% FP)
— miscounting starts out mostly correct, so there's less noise available for a
filter to remove.

## Dev batch results (13 FP + 15 TP, 3 PARSE_ERROR excluded)

| Metric | Value |
|---|---|
| Catch rate (FP correctly rejected) | 3/12 = **25.0%** |
| FN rate (TP wrongly rejected) | 2/13 = **15.4%** |

Unlike invention (20% catch, near-total rubber-stamping), Sonnet showed real
independent discrimination here — comparable to mischaracterization's natural
33.3% catch rate. Confirms the groundability hypothesis: counting, like color/
shape/material, is something Sonnet can independently verify from the image
rather than just deferring to Haiku's framing.

## Real scorer effect (27 touched rows, TRUE full-row methodology)

Uses each row's complete current-pipeline prediction (all labels), not just
isolated miscounting spans — the corrected methodology (see the
mischaracterization redo earlier the same night).

| Metric | Before | After | Delta |
|---|---|---|---|
| Cor | 0.4554 | 0.4730 | **+0.0177** |
| Cor+Lbl | 0.4342 | 0.4543 | **+0.0201** |

Positive, real discrimination confirmed — but small in magnitude compared to
mischaracterization's +0.0557/+0.0558 corrected per-row delta.

## Full-submission projection

236 miscounting flags across 220 of 1,201 TEST rows currently. Applying the
measured per-affected-row delta to all 220 affected rows:

| | Delta | Baseline → Projected |
|---|---|---|
| Cor | +0.0032 | 0.3824 → 0.3856 |
| Cor+Lbl | +0.0037 | 0.2897 → 0.2934 |

## Cost

28 Sonnet calls: input=109,633 tokens, output=3,013 tokens, **$0.2494**. Full
deployment on all 236 real flags would cost ~$1.27.

## Bottom line

Real, positive, and methodologically sound — but the projected gain
(+0.0037 Cor+Lbl) is roughly the same size as the numeral-relabeling fix's
confirmed real leaderboard gain (+0.0029), the smallest successful fix so far.
Worth having in the toolkit, but marginal relative to the mischaracterization
fix's much larger projected effect (+0.026). Priority should be deploying
mischaracterization first; miscounting is a small additive bonus, not a
priority on its own.

## Note: temperature pinning

Attempted to pin `temperature=0` on Sonnet calls before this test for
determinism -- the API rejected it: `'temperature' is deprecated for this
model` (claude-sonnet-5). Reverted; this model generation does not support
temperature control. Non-determinism in Sonnet's verdicts remains a real,
unresolved property of this pipeline step (see [[fix_sonnet_mischar_verification]]).
