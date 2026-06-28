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
    clean = re.sub(r"```json|```", "", raw_text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
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





"""import google.generativeai as genai
import base64
import json,re,os
from dotenv import load_dotenv
 # Load environment variables from .env file

def configure_gemini():
    load_dotenv() 
    api_key=os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ Error: GEMINI_API_KEY environment variable not set. Please set it before running the script.")
    genai.configure(api_key=api_key)
    
def load_image_as_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

def get_image_media_type(image_path):
    image_path_lower=image_path.lower()
    if image_path_lower.endswith(".jpg") or image_path_lower.endswith(".jpeg"):
        return "image/jpeg"
    elif image_path_lower.endswith("png"):
        return "image/png"  
    else:
        return "image/jpeg"  # Default to JPEG if unknown

def parse_llm_response(raw_text):
    clean=re.sub(r"```json|```", "",raw_text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON from response: {raw_text[:100]}")
        return []
    
def call_vision_llm(image_path,audit_prompt):
    configure_gemini()
    
    image_data=load_image_as_base64(image_path)
    media_type=get_image_media_type(image_path)
    
    model=genai.GenerativeModel("gemini-1.5-flash-8b")
    
    image_part={
        "inline_data":{
            "mime_type": media_type,
            "data": image_data
        }
    }
    
    response = model.generate_content([image_part, audit_prompt])
    raw_text=response.text
    return parse_llm_response(raw_text)
    """