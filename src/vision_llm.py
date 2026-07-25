import anthropic
import base64
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()

def load_image_as_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

def get_image_media_type(image_path):
    image_path_lower = image_path.lower()
    if image_path_lower.endswith(".jpg") or image_path_lower.endswith(".jpeg"):
        return "image/jpeg"
    elif image_path_lower.endswith(".png"):
        return "image/png"
    else:
        return "image/jpeg"

def parse_llm_response(raw_text):
    print(f"RAW FROM CLAUDE: {raw_text}")

    # Robust extraction: grab ONLY the first ```json ... ``` fenced block if
    # present, and ignore any explanatory prose Haiku adds before/after it.
    # A naive strip-and-parse breaks whenever the model reasons out loud
    # around the JSON, silently turning a valid answer into an empty list.
    fenced = re.search(r"```json\s*(.*?)```", raw_text, re.DOTALL)
    candidate = fenced.group(1).strip() if fenced else raw_text.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first top-level [...] array anywhere in the text.
    bracket_match = re.search(r"\[.*?\]", raw_text, re.DOTALL)
    if bracket_match:
        try:
            return json.loads(bracket_match.group(0))
        except json.JSONDecodeError:
            pass

    print(f"Warning: Could not parse JSON: {raw_text[:100]}")
    return []

def call_vision_llm(image_path, audit_prompt):
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    image_data = load_image_as_base64(image_path)
    media_type = get_image_media_type(image_path)
    
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        }
                    },
                    {
                        "type": "text",
                        "text": audit_prompt
                    }
                ]
            }
        ]
    )
    
    raw_text = response.content[0].text
    return parse_llm_response(raw_text),raw_text