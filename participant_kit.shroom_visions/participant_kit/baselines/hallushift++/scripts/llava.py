

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from types import SimpleNamespace


###################################
# LLaVA-v1.6-Mistral-7B Functions #
###################################

'''
LD_PRELOAD=/home/csavelli/shroom-vision/.conda/lib/libstdc++.so.6 \
/home/csavelli/shroom-vision/.conda/bin/python process_hidden_jsons.py \
  --jsonl hidden/shroom-vision.test.en.all_metadata.jsonl \
  --model llava \
  --debug
'''


LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"


def _load_image(image_path):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image at {image_path}: {e}")
        return None


def _build_inputs(processor, image, prompt, model):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        },
    ]
    chat_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return processor(images=image, text=chat_prompt, return_tensors="pt").to(model.device)


def _extract_hallushift_features(generated, decoded_text, tokenizer, model_type="llava"):
    import functions

    return (
        functions.plot_internal_state_2(generated, model_type=model_type)
        + functions.plot_internal_state_2(generated, state="attention", model_type=model_type)
        + functions.probability_function(generated)
        + functions.layer_prediction_consistency(generated, model_type=model_type)
        + functions.attention_concentration_features(generated)
        + functions.perplexity_confidence_features(generated)
        + functions.token_repetition_novelty_features(decoded_text, tokenizer)
    )


from transformers import LlavaForConditionalGeneration, LlavaProcessor
import torch

LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

def llava_loading(custom_hf_cache_dir=None):
    model_kwargs = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }

    if custom_hf_cache_dir is not None:
        model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_MODEL_ID,
            cache_dir=custom_hf_cache_dir,
            **model_kwargs,
        )
        processor = LlavaProcessor.from_pretrained(
            LLAVA_MODEL_ID,
            cache_dir=custom_hf_cache_dir,
            use_fast=False,
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_MODEL_ID,
            **model_kwargs,
        )
        processor = LlavaProcessor.from_pretrained(
            LLAVA_MODEL_ID,
            use_fast=False,
        )

    model.eval()
    return model, processor


def llava_hallushift_inference(model, processor, prompt, image_path, generation_kwargs=None):
    if generation_kwargs is None:
        generation_kwargs = {}
    else:
        generation_kwargs = dict(generation_kwargs)

    image = _load_image(image_path)
    if image is None:
        return None

    inputs = _build_inputs(processor, image, prompt, model)
    prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0

    generated = model.generate(
        **inputs,
        max_new_tokens=generation_kwargs.pop("max_new_tokens", 64),
        do_sample=generation_kwargs.pop("do_sample", False),
        pad_token_id=generation_kwargs.pop(
            "pad_token_id",
            processor.tokenizer.eos_token_id if hasattr(processor, "tokenizer") else None,
        ),
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_attentions=True,
        output_logits=True,
        **generation_kwargs,
    )

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    decoded = tokenizer.decode(generated.sequences[0][prompt_len:], skip_special_tokens=True).strip()
    features = _extract_hallushift_features(generated, decoded, tokenizer, model_type="llava")

    return {
        "response": decoded,
        "features": features,
        "generated": generated,
    }


def llava_inference(model, processor, prompt, image_path, generation_kwargs=None):
    result = llava_hallushift_inference(model, processor, prompt, image_path, generation_kwargs=generation_kwargs)
    if result is None:
        return None
    return result["response"]


def llava_features_from_response(model, processor, prompt, image_path, response_text, include_features=True):
    """Compute HalluShift features for an already-generated response.

    This runs a teacher-forcing forward pass using the provided `response_text`
    (no sampling), collects hidden states/attentions/logits for the decoder and
    reshapes them into the same structure expected by the existing feature
    extraction helpers in `functions.py`.

    Returns a dict with keys: `response` (the input response_text),
    `features` (list of 74 floats), and `generated` (a lightweight object
    with `.hidden_states`, `.attentions`, `.logits`, `.sequences`).
    """
    image = _load_image(image_path)
    if image is None:
        return None

    inputs = _build_inputs(processor, image, prompt, model)

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    resp_ids = tokenizer.encode(response_text, add_special_tokens=False)
    if len(resp_ids) == 0:
        # nothing to score
        return {"response": response_text, "features": [0.0] * 74, "generated": None}

    device = inputs["input_ids"].device
    response_ids = torch.tensor([resp_ids], device=device)

    prompt_ids = inputs["input_ids"]
    prompt_len = prompt_ids.shape[1]
    response_len = response_ids.shape[1]

    full_input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    if "attention_mask" in inputs:
        full_attention_mask = torch.cat(
            [inputs["attention_mask"], torch.ones((1, response_len), dtype=inputs["attention_mask"].dtype, device=device)],
            dim=1,
        )
    else:
        full_attention_mask = torch.ones_like(full_input_ids, device=device)

    full_labels = torch.cat(
        [torch.full((1, prompt_len), -100, dtype=torch.long, device=device), response_ids],
        dim=1,
    )

    model_inputs = dict(inputs)
    model_inputs["input_ids"] = full_input_ids
    model_inputs["attention_mask"] = full_attention_mask
    model_inputs["labels"] = full_labels
    model_inputs["output_hidden_states"] = True
    model_inputs["output_attentions"] = True
    model_inputs["return_dict"] = True

    with torch.no_grad():
        outputs = model(**model_inputs)

    # Prefer decoder-side attributes when available
    decoder_hidden = getattr(outputs, "decoder_hidden_states", None) or getattr(outputs, "hidden_states", None)
    decoder_attns = getattr(outputs, "decoder_attentions", None) or getattr(outputs, "attentions", None)

    sequences = full_input_ids

    # Build per-time-step tuples of layer tensors to match the structure used by
    # the original generation-based outputs (a list over time-steps of tuples).
    hidden_states_per_step = []
    if decoder_hidden is not None:
        for t in range(response_len):
            token_idx = prompt_len + t
            per_layer = tuple(
                layer[:, token_idx, :] if hasattr(layer, "dim") and layer.dim() == 3 else layer
                for layer in decoder_hidden
            )
            hidden_states_per_step.append(per_layer)

    attentions_per_step = []
    if decoder_attns is not None:
        for t in range(response_len):
            token_idx = prompt_len + t
            per_layer = []
            for att in decoder_attns:
                try:
                    if hasattr(att, "dim") and att.dim() == 4:
                        # att: (batch, heads, seq_len, seq_len) -> keep 4D shape per generated token query
                        per_layer.append(att[:, :, token_idx:token_idx + 1, :])
                    else:
                        per_layer.append(att)
                except Exception:
                    per_layer.append(att)
            attentions_per_step.append(tuple(per_layer))

    logits = getattr(outputs, "logits", None)
    if logits is not None and hasattr(logits, "dim") and logits.dim() == 3:
        logits = tuple(logits[:, prompt_len + t - 1, :] for t in range(response_len))

    generated_like = SimpleNamespace()
    generated_like.hidden_states = hidden_states_per_step
    generated_like.attentions = attentions_per_step
    generated_like.logits = logits
    generated_like.sequences = sequences

    features = _extract_hallushift_features(generated_like, response_text, tokenizer, model_type="llava") if include_features else []

    return {"response": response_text, "features": features, "generated": generated_like}