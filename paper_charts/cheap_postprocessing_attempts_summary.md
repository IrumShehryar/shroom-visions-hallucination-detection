# Discussion / Future Work (main body — brief)

*Note: appendix lettering below (A-E) reflects this document's internal order only.
Adjust to fit the paper's actual appendix numbering (e.g. if a separate Prompt
Development appendix precedes these).*

No single method closed the gap between the current pipeline and a meaningfully
higher score, and the error surface does not reduce to one problem. Across five
targeted investigations — three probabilistic output filters, one deterministic
relabeling rule, a systematic confusion-matrix analysis, a mechanism test for the
model's recall gap, and a feasibility study for retrieval-augmented (RAG)
verification — only the deterministic rule delivered a validated, leaderboard-
confirmed improvement (Cor+Lbl +1.0%), and it addressed a narrow, syntactically
identifiable slice of the error surface. The remaining errors split across at
least three qualitatively distinct problems, each requiring a different kind of
fix: unreliable model self-confidence that does not calibrate portably across
batches (Appendix A), a span-boundary granularity mismatch between what the model
naturally flags and what gold annotations isolate — not an attention or coverage
failure (Appendix D), and a mix of purely visual and externally-verifiable factual
content within the invention category that no single verification method covers
end to end (Appendix E). No post-processing rule, prompt tweak, or single external
tool addresses more than one of these at once.

On the last point specifically: a feasibility analysis of retrieval-augmented
(RAG) verification for invention-category errors found that of 154 gold invention
spans in the 286-sample dev set, **26.6%** are checkable against external text
sources — either directly (14.9%, pure factual/scientific claims) or as supporting
reference material for image-grounded verification (11.7%, named-entity
identification such as species or brand). At the response level, **51.1%** of
responses containing an invention error include at least one RAG-helpable claim.
A further **23.4%** of spans are annotation-granularity fragments rather than
coherent claims (e.g. `"a "`, `"the "`), an intrinsic ceiling on any claim-level
verification method, RAG included. See **Appendix E** for the full category
breakdown, the response-level re-aggregation, and methodology.

---

# Appendix A: Rejected Post-Processing Interventions

**Motivation.** The final test submission scored Cor = 0.3816, Cor+Lbl = 0.2868 on
the official leaderboard. Because `scorer.py`'s `score_cor_lbl` averages per-label
Spearman correlation over the *union* of labels present in gold and prediction for
each row, any predicted label absent from a row's gold hard-zeros that label's term
— so wrong-type flags are disproportionately costly to Cor+Lbl specifically. Offline
analysis of the current pipeline against 286 labeled train rows (`scoring_run/`)
found a 60.4% false-positive rate on genuinely gold-clean rows (mean confidence
0.83), with `mischaracterization` responsible for 54% of those false positives and
the worst per-label IoU (0.16) of any category. This motivated four candidate
post-processing filters, tested with zero additional API calls against already-cached
predictions and the official scorer, then checked against a genuinely fresh,
never-tuned-on held-out batch (n=39, sampled from train-en rows excluded from all
prior tuning sets, run live through the current pipeline).

## A.1 Results

All scores computed with the official `scorer.py` (`score_cor`, `score_cor_lbl`,
`score_iou`), on the labeled train split (not the real test set — these are offline
proxy measurements).

| Filter | Dev Cor (n=286) | Dev Cor+Lbl | Dev IoU | Held-out Cor (n=39) | Held-out Cor+Lbl | Held-out IoU |
|---|---|---|---|---|---|---|
| Baseline (no filter) | 0.4313 | 0.3796 | 0.4294 | 0.3252 | 0.2684 | 0.2707 |
| Confidence threshold (mischaracterization ≥ 0.92) | 0.4636 (+7.5%) | 0.4213 (+11.0%) | 0.4671 (+8.8%) | 0.2757 (−15.2%) | 0.2316 (−13.7%) | 0.2418 (−10.7%) |
| Hedge-language suppression (drop mischar. flags whose `reason` contains hedge phrasing) | 0.4556 (+5.6%) | 0.4041 (+6.5%) | 0.4508 (+5.0%) | 0.3026 (−7.0%) | 0.2561 (−4.6%) | 0.2598 (−4.0%) |
| Negation (sentence-level: drop flags whose containing sentence has any negation word) | 0.4325 (+0.3%) | 0.3866 (+1.8%) | 0.4354 (+1.4%) | 0.3048 (−6.3%) | 0.2616 (−2.5%) | 0.2582 (−4.6%) |

A best-combination sweep (mischaracterization ≥ 0.92 + invention ≥ 0.5) reached
Cor 0.4706 / Cor+Lbl 0.4283 / IoU 0.4741 on the dev set — the single best offline
number found in this investigation — and was not re-tested on held-out data once
the pattern below became clear.

A separate sanity check against 226 rows from an older-prompt scoring run
(`old_prompt_v1_scoring_run/`, non-overlapping with the 286-row dev set) showed the
same confidence-threshold filter *improving* results even more sharply there
(Cor 0.3759→0.4475, Cor+Lbl 0.3264→0.4147, IoU 0.3221→0.4041) — corroboration that
turned out to be misleading once checked against genuinely fresh data.

**Every filter helps the dev set and hurts the held-out set.** This is the paper's
relevant finding: three structurally different filtering strategies (a numeric
confidence cutoff, a linguistic hedge-word cue, a sentence-level negation cue) each
independently passed offline validation on 226-286 labeled rows and each reversed on
a small, genuinely fresh batch. Root-cause check on the confidence-threshold filter:
in the held-out batch, correctly-matching mischaracterization flags clustered at
0.75-0.90 confidence (21 of them) while only 2 wrong-type flags existed to filter out
— the 0.92 cutoff was removing signal, not noise, in that batch.

## A.2 A precise mechanism, structurally disproved

A fourth, narrower hypothesis was tested: that Haiku extracts a bare noun from a
negated clause in the response (e.g. flagging `"cucumber"` inside `"...not
elongated...like a typical cucumber would be"`) and disputes it as an unqualified
claim. Rather than validate this statistically, we searched the full 325-sample pool
(dev + held-out) for the literal pattern — a flagged span exactly matching a noun
phrase immediately preceded by "not"/"no"/"isn't"/"doesn't have" (± an article), with
no negation word inside the span itself. **Zero matches.** The two examples that had
motivated the hypothesis turned out to be the same underlying image
(`cucumber_Lemon_cucumber_Cucumber-Crystal-Lemon.jpg`, a real "lemon cucumber"
cultivar) sampled into two different prompt/response pairs, not independent evidence
— and on closer reading, the actual mechanism was a different one (an over-broad
comparison/simile flag, not negation-parsing), too narrow and non-independent to
generalize. This is included here as a methodological example: **structural
validation against the literal proposed mechanism is a strictly stronger check than
an aggregate score delta**, and it caught a false lead that a favorable n=39 metric
swing could not have distinguished from a real effect.

## A.3 Takeaway

None of the four candidate filters were adopted. The recurring failure pattern
across independently-designed heuristics suggests the confidence/language signal
Haiku emits for `mischaracterization` does not calibrate portably across sampling
batches at the scale tested here (n≈40 held-out), rather than that any one heuristic
was specifically flawed. A separate attempt to use a real OCR engine (EasyOCR) to
corroborate or suppress OCR-category flags was tried earlier and also did not
improve scores (a 105-sample manual breakdown found EasyOCR confirmed only 22.4% of
true-positive OCR disputes, though it did align with 60.0% of false-positive OCR
flags — an asymmetry that made it a poor blunt gate). Figure: `threshold_filter_dev_vs_heldout.png`.

---

# Appendix B: Adopted Fix — Numeral-Based Relabeling

**Mechanism.** `score_cor_lbl` computes a separate Spearman correlation per label,
averaged over the union of labels present in gold or prediction for that row. A
span mislabeled `mischaracterization` when gold says `miscounting` therefore costs
*two* zero-scored label channels — the wrong label predicted, the right label
missing — a penalty structure IoU (pure character-overlap, label-blind) cannot
capture. This is why an earlier, IoU-only test of the same relabeling idea (see
project memory) measured as net negative/negligible, while re-testing against
Cor+Lbl specifically revealed a real effect.

**Rule.** Relabel a `mischaracterization` flag to `miscounting` when its span is
short (≤3 tokens) and centers on a bare number, *excluding* percentages (`"75%"`,
`"80 percent"`), clock-time format (`"10:25"`), and referential "one" not
immediately followed by a plural noun (`"the white one"`, the North-Pole/Pole pun
`"one Pole"`) — these three exclusions were added after the first version of the
rule was checked against the real, already-submitted test predictions and found to
mislabel exactly these three patterns (6 of 25 raw triggers, 24%).

## B.1 Validation (same held-out discipline as Appendix A)

| Dataset | Baseline Cor+Lbl | Fixed Cor+Lbl | Note |
|---|---|---|---|
| Dev (n=286) | 0.3796 | 0.3892 (+2.5%) | Cor stays flat (0.4313→0.4318) — the label-only-fix signature |
| Held-out (n=39) | 0.2684 | 0.2684 (unchanged) | 0 triggers in this batch (base rate ~3%, expected ~1.2) — untested, not confirmed or denied |
| Old-prompt (n=226, free) | 0.3264 | 0.3307 (+1.3%) | Both triggers exact-match gold position |
| Real test submission (n=1,201) | — | — | 19 triggers across 17 rows, all manually verified as genuine object counts (wheels, tines, holes, screws, claws, buckles, columns, horns, beads) |

**Confusion-matrix confirmation**, on the combined 286+226+39=551 sample pool:
the target cell (gold `miscounting` → predicted `mischaracterization`) dropped from
21 to 5 instances (−76%) after the fix; the reverse cell (gold `mischaracterization`
→ predicted `miscounting`) was essentially unchanged (2→3), as expected for a
one-directional rule.

**Caveat on precision, found via gold-label inspection, not just the aggregate
score:** checking exact character-position overlap (not just "does this row have
the label somewhere") on the 9 dev-set triggers showed 5/9 land cleanly on a gold
`miscounting` span, and 2/9 land on a span that *also* carries a gold
`mischaracterization` annotation from a different human annotator (e.g. `"four"` in
`train-en-11143` is independently double-labeled `mischaracterization` and
`miscounting` by different annotators) — genuine annotator disagreement rather than
the rule being wrong. The remaining 2/9 had no exact-position gold overlap at all.

**Status: merged into `src/span_mapper.py` (applied inside `map_spans_to_characters`,
so every caller gets it automatically) and applied to the test submission
(`submission/en.jsonl`, regenerated from the already-cached raw predictions — no
new API calls). Prior submission preserved at `submission/en.jsonl.pre_numeral_fix_backup`.**

**Confirmed on the real leaderboard** (submission "Haiku_vision_verifier_v2_numeral_fix"):
Cor 0.3816 → 0.3824 (+0.2%), **Cor+Lbl 0.2868 → 0.2897 (+1.0%)**. Matches the
pre-submission estimate (extrapolated from the dev-set delta, projected to land
around 0.289-0.293) and confirms the mechanistic signature one more time: Cor+Lbl
moved ~5x more than Cor in relative terms, consistent with a fix that corrects only
label identity, not span position. This is the one candidate fix out of five tried
(confidence threshold, hedge-language, negation, OCR-corroboration, numeral
relabeling) that survived every validation stage and delivered a real, positive,
platform-confirmed gain.

---

# Appendix C: Confusion-Matrix Analysis

After the numeral fix, a full 5×5 gold-label × predicted-label confusion matrix was
built on the same 551-sample combined pool (any character overlap counts) to check
whether other systematic mislabeling patterns exist.

**Headline finding: detection failure dominates over labeling error in every
category.** Missed spans (zero overlap with any prediction) outnumber wrong-label
spans 3-10x across the board:

| Gold category | Missed | Wrong label | Correct | n |
|---|---|---|---|---|
| invention | 73.6% | 14.9% | 11.4% | 402 |
| mischaracterization | 74.1% | 7.6% | 18.2% | 510 |
| miscounting | 50.5% | 6.3% | 43.2% | 206 |
| OCR | 22.7% | 16.7% | 60.7% | 150 |
| other | 56.8% | 10.2% | 33.0% | 88 |

This caps how much post-processing (relabeling what the model *did* flag) can ever
recover — most of the error budget is spans the model never flagged at all, which
no relabeling rule can create from nothing.

**The largest remaining labeling confusion:** invention↔mischaracterization,
81 total instances (51 invention→mischaracterization, 30 the reverse) — 3.5x larger
than the miscounting↔mischaracterization confusion was before the fix (23). Per-label
Cor sub-scores confirm these are the two weakest categories (mischaracterization
0.550, invention 0.723, vs. 0.88-0.90 for the other three).

**This confusion was investigated for a structural fix and found not to have one.**
Both directions share the same surface shape — a short noun phrase naming what
something is ("handle", "container", "Fossil", "Kune Kune pig", "pine",
"Gray Fox") — tracing back to the prompt's own category definitions, which overlap
on naming (mischaracterization's definition explicitly includes "name" as a
disputable property; invention's definition explicitly includes "invented names for
things"). Unlike the numeral signal (unidirectional — a digit always means
"counting," so the correction always goes one way), the short-noun-phrase signal is
**direction-symmetric**: among short gold-invention spans, 16.1% are mislabeled
mischaracterization vs. only 8.9% labeled correctly; among short
gold-mischaracterization spans, 17.2% are labeled correctly vs. 6.0% mislabeled
invention. Neither direction dominates enough to support a blanket relabel rule —
doing so would trade one error type for another with no net gain. This is presented
as a deliberate negative result: a category boundary that is ambiguous in the task
definition itself, not fixable by post-processing, in contrast to the miscounting
case which had an unambiguous syntactic tell.

---

# Appendix D: Recall-Gap Mechanism Test

The confusion-matrix analysis (Appendix C) found the two weakest categories
(invention, mischaracterization) are missed entirely (zero overlapping prediction)
73-74% of the time, far outweighing mislabeling (8-15%). Response length and
position within the response both correlate strongly and monotonically with miss
rate (62.9%→85.8% across length quartiles; 64.7%→87.1% across position quartiles),
consistent with an attention-decay hypothesis — the model catching less as a
response gets longer or as content appears later in it.

**A cheap mechanism test (15 real API calls, ~$1-2) disconfirmed the strong version
of this hypothesis before any prompt restructuring was attempted.** 15 known-missed,
long-response, late-position gold spans were re-audited with the response split in
half and only the half containing the target fed to the model, as if it were the
entire response. Only 2/15 (13.3%) were caught exactly. But a secondary check — does
the model flag *anything in the same sentence* as the target, even without exact
overlap — showed **15/15 (100%)**, both in the isolated-half re-run and in the 4 of
these cases whose original full-length response was already cached.

**The model is attending to and disputing the correct area of content in every case
checked; the "miss" is a boundary mismatch** between the phrase the model naturally
flags and a narrower single-word sub-span some gold annotations pick out (e.g. gold
wants just `"not"` or `"long"`; the model flags `"shorter, more compact snouts"` or
`"long barrel"` — same dispute, different boundary). This matches and reinforces an
earlier, independently-failed attempt (POS-tag-based span-narrowing for
mischaracterization, Appendix B's precursor, documented in project memory, which
measurably regressed IoU) — narrowing a correctly-identified broad span down to a
specific word appears to be a genuinely hard problem with the current architecture,
not a prompt-wording fix. Both claim-enumeration prompt restructuring and additional
worked examples were de-prioritized on this evidence, since neither targets a
boundary-granularity problem.

---

# Appendix E: RAG Feasibility Analysis for Invention-Category Errors

## E.1 Motivation

A collaborator proposed that invention errors split into two sub-types — fabricated
explanations checkable against external text sources (RAG-suitable) vs. fabricated
visual details checkable only from the image (RAG-unsuitable) — matching the
architecture used by UCSC's top-ranked Mu-SHROOM submission. To test this against
our own data rather than assume it, all 154 gold invention spans in the 286-sample
dev set were read in full context and classified into five categories.

## E.2 Span-level breakdown (154 spans)

| Category | Span-level count | Share | RAG help? |
|---|---|---|---|
| Pure visual/existence (object presence, spatial layout, color, texture) | 71 | 46.1% | No |
| Annotation fragments (partial phrases, connectors — not coherent claims) | 36 | 23.4% | N/A |
| Pure text-fact/scientific claim (true/false independent of the image) | 23 | 14.9% | Yes, directly |
| Hybrid: named-entity ID (species/breed/brand/event/location) | 18 | 11.7% | Yes, but still needs image-grounding |
| Speculative/unverifiable inference (opinion-like, no fact to check) | 6 | 3.9% | No |

**Span-level RAG-relevant total: 41/154 = 26.6%.** This refines the collaborator's
framing: the clean "text-fact" bucket alone is only 14.9%; the larger RAG-relevant
share comes from the **hybrid** bucket (species/breed/brand/location identification,
e.g. "Chioggia carrots," "Muscovy Duck," "Auckland," "Devonport"), where RAG would
supply a reference description for the VLM to cross-check against the image, not
replace image-grounding. Two caveats found empirically, not assumed: 46.1% of spans
are irreducibly visual and untouchable by any text-retrieval approach, and a
striking 23.4% are annotation-granularity fragments (`"a "`, `"the "`, single
letters) that no claim-level method — RAG or otherwise — can act on, since there is
no complete claim to verify. This fragment rate is the same phenomenon behind the
span-boundary mechanism finding in Appendix D.

## E.3 Response-level breakdown (45 responses)

Re-aggregated at the response level (45 distinct responses contain a gold invention
span, vs. 154 individual spans — one verbose or heavily-fragmented response can
otherwise dominate the span-level tally):

| | Count | Share |
|---|---|---|
| Responses with ≥1 RAG-helpable (text-fact or hybrid) span | 23 | **51.1%** |
| Responses that are entirely visual/fragment/speculative | 22 | 48.9% |

## E.4 Category-mix per response

The full breakdown of which category *combinations* occur within a single response
— this is what drives the divergence between the span-level (26.6%) and
response-level (51.1%) numbers:

| Response's category mix | Count |
|---|---|
| Visual-only | 8 |
| Hybrid-only | 7 |
| Fragment + visual | 6 |
| Fragment-only | 5 |
| Fragment + text-fact + visual (mixed) | 4 |
| Text-fact-only | 4 |
| Hybrid + fragment | 2 |
| Hybrid + text-fact | 2 |
| Fragment + speculative + visual | 2 |
| Speculative-only | 1 |
| Hybrid + fragment + visual | 1 |
| Speculative + text-fact + visual | 1 |
| Speculative + text-fact | 1 |
| Text-fact + visual | 1 |

## E.5 Illustrative examples

Three concrete cases, to ground the percentages above:

- **`train-en-903` (crab legs, "fragment + text-fact + visual" mix).** The response
  argues a crab has eight legs, not the ten the prompt asks about. Gold marks
  `"the typical number of legs for a decapod"` and `"Crabs are a type of decapod...
  generally possess eight legs in addition to two claws"` as invention — genuine
  biological facts, checkable against any zoology reference independent of this
  specific photo. The same response also has several short fragment-annotated
  spans (`"which is a common misconception"`, mid-sentence pieces) that are not
  independently RAG-actionable. This response counts as RAG-helpable at the
  response level, even though most of its individual spans are not.

- **`train-en-10858` (immersion blender, "visual-only").** The response repeatedly
  claims the blender has a `"container"` — seven separate gold-annotated
  occurrences of the same word across the response, all disputing whether that
  specific part exists on this specific object. Nothing in an external knowledge
  source resolves whether *this* blender has *this* part visible in *this* photo —
  it's answerable only from the pixels. This single response contributes 7 of the
  71 visual-only spans in the span-level tally, which is why the response-level
  re-aggregation (E.3) is the fairer comparison.

- **`train-en-2356` (pig breed, "hybrid-only").** The response guesses the pig is a
  `"Kune Kune pig"` (or alternatively Berkshire, Hampshire, Duroc). Gold disputes
  this identification. Resolving it requires *both* the image (what does this
  specific pig actually look like) *and* external reference knowledge (what
  distinguishes a Kune Kune from a Mangalitsa or Berkshire) — the textbook case for
  a RAG+VLM hybrid: RAG supplies the breed-distinguishing description, the VLM
  performs the visual match. Pure text retrieval alone cannot resolve this; pure
  image-grounding alone (the current pipeline) also struggles, since it has no
  reference description of what a Kune Kune actually looks like beyond what's in
  Haiku's training data.

## E.6 Recommendation

A hybrid RAG+VLM architecture is evidence-supported for roughly a third of
invention errors by volume and could plausibly touch just over half of affected
responses, concentrated in the named-entity/species/brand identification sub-type
specifically — not general invention detection. This is a substantially larger
engineering effort than anything else in this document (retrieval infrastructure, a
knowledge source, and per-claim routing logic between RAG and VLM verification) and
was not built or tested end-to-end; this section documents the evidence-based case
for scoping it, not a validated result.
