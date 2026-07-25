"""
Narrow post-processing rule: relabel a mischaracterization flag as
miscounting when its span_text is short and centers on a bare number
(digit or number word) that is quantifying a countable object -- e.g.
"four prongs", "two humps", "three". Excludes percentages, clock times,
and referential "one" ("the white one"), which surfaced as real
false-trigger patterns when checked against the actual test submission
(see project memory: attempt_confidence_threshold_filter's sibling note on
this rule, and paper_charts/cheap_postprocessing_attempts_summary.md).

Mechanistically distinct from the earlier IoU-based test of the same idea
(see project memory: net negative/negligible on IoU). score_cor_lbl computes
a separate correlation per label and averages over the union of labels in
gold+pred, so a mislabeled span costs TWO zero-scored label channels
(the wrong one predicted, the right one missing) in a way IoU -- pure
char-overlap, label-blind -- cannot capture. This targets Cor+Lbl
specifically, not IoU (the competition this year scores Cor and Cor+Lbl
only, no IoU).
"""
import re

from src.span_mapper import NUMBER_WORDS, _tokenize_words

MAX_TOKENS = 3

PERCENT_RE = re.compile(r"%|percent", re.IGNORECASE)
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")


def _is_excluded(span_text, toks):
    if PERCENT_RE.search(span_text):
        return True
    if TIME_RE.search(span_text):
        return True
    # "one" is a genuine quantifier when it leads a noun phrase ("one blue
    # dumpster" = a count of 1). It's referential, not a count, when it's
    # the trailing pronoun in "[adjective] one" ("the white one") or when
    # it's immediately followed by what looks like a proper noun / pun
    # ("one Pole"). Only exclude those two shapes, not "one" in general --
    # blanket-excluding every multi-token "one" span also throws out
    # legitimate quantifier uses like "one blue dumpster".
    for i, t in enumerate(toks):
        if t.lower() != "one" and t != "1":
            continue
        is_last = i == len(toks) - 1 and len(toks) > 1
        next_tok = toks[i + 1] if i + 1 < len(toks) else None
        followed_by_capitalized = bool(next_tok) and next_tok[:1].isupper()
        if is_last or followed_by_capitalized:
            return True
    return False


def relabel_miscounting(llm_output):
    """Returns a new list; relabels qualifying mischaracterization entries
    to miscounting. Does not mutate the input."""
    out = []
    for entry in llm_output:
        e = dict(entry)
        if str(e.get("label", "")).lower() == "mischaracterization":
            span = e.get("span_text", "")
            toks = _tokenize_words(span)
            has_num = any(t.isdigit() or t.lower() in NUMBER_WORDS for t in toks)
            if has_num and 0 < len(toks) <= MAX_TOKENS and not _is_excluded(span, toks):
                e["label"] = "miscounting"
        out.append(e)
    return out
