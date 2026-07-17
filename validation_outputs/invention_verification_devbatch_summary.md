# Invention-Label Sonnet Verification — Bounded Dev-Batch Check

Generated from `verify_invention_devbatch_results.json` (20-sample curated
batch: 10 known FP + 10 known TP, drawn from the same dev-cache pool used for
the mischaracterization work, seed=4242). This was a deliberately bounded
check, not a full dev+held-out replication of the mischaracterization
pipeline — see conversation for rationale.

## Natural population (free, no API calls)

Every invention flag the current pipeline emits, across the 325 gold-checked
dev+fresh rows:

| | count | % |
|---|---|---|
| Total invention flags | 80 | 100% |
| TP (overlaps gold invention span) | 20 | 25.0% |
| FP (no gold invention overlap) | 60 | 75.0% |

Even noisier than mischaracterization's natural 68% FP rate.

## Dev batch results (10 FP + 10 TP)

| Metric | Value |
|---|---|
| Catch rate (FP correctly rejected) | 2/10 = **20%** |
| FN rate (TP wrongly rejected) | 1/10 = **10%** |

Sonnet mostly confirmed Haiku's invention flags regardless of correctness —
e.g. `train-en-1766`'s fabricated species ID ("striped skunk, *Mephitis
mephitis*") and `train-en-2083`'s "lost its wing" claim were both known false
positives that Sonnet confirmed instead of rejecting. The two flags it did
correctly reject were general factual claims ("grapefruits are typically
round," "blueberries ripen to pink/red"), not image-specific existence
claims.

**Working hypothesis**: many invention flags hinge on fine-grained
identification (species names, scientific binomials, brand/model guesses)
that Sonnet can't reliably verify against the image any better than Haiku
could — so it defaults toward agreeing with Haiku's conclusion rather than
independently re-checking. Mischaracterization claims (color, shape, count,
material) are more directly groundable visually, which likely explains why
that category showed a real signal and this one doesn't.

## Real scorer effect (19 touched rows, invention spans only)

| Metric | Before | After | Delta |
|---|---|---|---|
| Cor | 0.1552 | 0.1409 | **−0.0143** |
| Cor+Lbl | 0.1233 | 0.1002 | **−0.0230** |

**Net negative.** Unlike mischaracterization, applying the same drop-on-reject
filter to invention flags would hurt, not help.

## Cost

20 Sonnet calls: input=48,497 tokens, output=1,797 tokens, **$0.1150**.

## Conclusion

The mischaracterization-only scope for `ENABLE_SONNET_VERIFICATION`
(`src/verify_mischaracterization.py`) is correct. Invention was tested, not
just assumed unpromising, and the bounded check showed a clean negative
result -- not adopted. If ever revisited, the identification-vs-groundable
distinction above would be the first thing to test a fix against (e.g.
scoping verification to invention flags that are NOT taxonomic/brand/model
claims).
