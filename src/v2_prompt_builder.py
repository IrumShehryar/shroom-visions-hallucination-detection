def build_audit_prompt(prompt_text, response_text, filename_hint):
    role = """You are a precise hallucination span detector analyzing
outputs from vision-language models.

You will be given:
- An image
- A prompt that was asked about the image
- A response that a vision-language model generated

Your job is to find specific words or phrases in the response
that do not precisely match what is visible in the image.

IMPORTANT INSTRUCTIONS:
- Even if the response is broadly correct, individual words
  or phrases may be inaccurate. Find those specific words.
- Do not evaluate the response as a whole.
- Evaluate each factual claim individually against the image.
- A response can be mostly right but contain specific wrong
  details — your job is to find those details.
- Pay attention to: specific names, varieties, brands, colors,
  quantities, spatial relationships, and visible text."""

    # ONLY CHANGE vs. original: expanded INVENTION definition + one worked
    # example, targeting the "missed abstract/unverifiable claim" pattern.
    # Nothing else in this prompt differs from the final reverted version.
    categories = """There are five hallucination categories:
        INVENTION: The response mentions something that does not
        exist in the image at all, OR states a fact, explanation,
        or piece of terminology that cannot be verified from the
        image and is not something a careful observer could
        determine just by looking. This includes invented locations,
        objects, invented technical/scientific explanations, and
        invented names for things.
        Example (spatial): Claiming the image shows Auckland New
        Zealand when no location markers are visible.
        Example (fabricated explanation): Claiming a bird's unusual
        white coloring is "due to a genetic condition called leucism"
        when nothing about genetics can be determined from a photo —
        this is a confidently-stated but unverifiable claim, which
        counts as invention even though it doesn't contradict anything
        visible.

        MISCHARACTERIZATION: Something real in the image is
        described incorrectly. The object exists but a property
        is wrong — color, material, direction, name.
        Example: Saying cables are attached to the ceiling when
        they are attached to the wall.

        MISCOUNTING: The response states a wrong quantity of
        something visible in the image.
        Example: Saying "four legs" when the animal has three.

        OCR: The response misreads text that is visibly
        written in the image.
        Example: Reading a slide as "sign on" when it says "log in".
        When flagging spans follow these rules:

        OTHERS: Any other hallucination that does not fit the above categories.
        This includes cases where the response fails to actually answer
        the prompt -- for example, if the response simply repeats or
        echoes the question back (e.g., "Human: what is X?") instead of
        providing a real answer. In such cases, flag the entire
        non-answering response as "other" with high confidence.
    """
    span_rules = """SPAN EXTRACTION RULES:

    RULE 1 - MISCOUNTING: You MUST extract ONLY the exact number word or numeral. Do NOT include the noun being counted.
    * CORRECT: "four"
    * INCORRECT: "four legs"
    * CORRECT: "2"
    * INCORRECT: "2 front legs"

    RULE 2 - MISCHARACTERIZATION: Extract ONLY the specific word or short property phrase that is false. Do NOT return full clauses or explanations.
    * CORRECT: "ceiling" or "the ceiling" (if it's attached to a wall)
    * INCORRECT: "cables attached to the ceiling"

    CRITICAL MANDATE: The "span_text" must be the shortest possible verbatim substring from the RESPONSE. If you include extra correct words surrounding the error, the evaluation system will penalize it with a 0 score."""

    ocr_guide = """If the prompt asks about text visible in the image:
        - Read the text in the image very carefully before
        evaluating the response
        - Compare word by word against what the response claims
        - Flag any word that differs from what the image shows
        -If the image text is too blurry to read confidently,
        do not flag OCR errors — assign low confidence only
        """

    confidence_guide = """For each flagged span assign a confidence score between 0 and 1:

        0.8 - 1.0: You are certain this is wrong. The image clearly
        contradicts this claim. Any reasonable person
        looking at the image would agree.

        0.5 - 0.7: You are fairly confident this is wrong but there
        is some ambiguity. The image suggests it is wrong
        but interpretation is needed.

        0.2 - 0.4: You think this might be wrong but you are not sure.
        The claim is questionable but the image does not
        clearly contradict it.

        If you are genuinely unsure whether something is wrong,
        assign low confidence rather than not flagging it.
        If nothing is clearly wrong, return an empty list.
        Be conservative — do not flag things unless you have
        clear visual evidence they are wrong.
        If you are not completely certain something is wrong based on
        clear visual evidence, do NOT flag it.
        Prefer returning empty [] over uncertain flags.
    """

    output_format = """Return your answer as a JSON array only. No other markdown formatting, no conversational text.

    Before writing each "span_text", verify: Is this the absolute minimum string required to isolate the error? For numbers, is it just the digit/number word?

    CRITICAL: Your entire response must be ONLY the JSON array -- nothing
    before it, nothing after it. Do not think out loud, do not write
    "wait, let me reconsider" or any re-examination in your output. If
    you are uncertain whether something should be flagged, resolve that
    uncertainty silently before writing the JSON -- only include a flag
    in the JSON if you are confident about it at the moment you write it.
    Never include a flag you go on to contradict or retract afterward.

    Each entry must have:
    - "span_text": the shortest exact substring from the response
    - "label": one of invention, mischaracterization, miscounting, OCR, other
    - "prob": confidence score between 0 and 1
    - "reason": one sentence explanation
    """
    input_section = f"""
    FILENAME HINT: {filename_hint}
    PROMPT: {prompt_text}
    RESPONSE: {response_text}
    """
    full_prompt = f"{role}\n\n{categories}\n\n{span_rules}\n\n{ocr_guide}\n\n{confidence_guide}\n\n{output_format}\n\n{input_section}"
    return full_prompt
