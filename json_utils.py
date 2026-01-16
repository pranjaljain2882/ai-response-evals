import json
import re

def extract_json(text: str) -> dict:
    """
    Extract the first JSON object found in a string.
    """
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in LLM output")

    json_str = match.group(0)
    return json.loads(json_str)
