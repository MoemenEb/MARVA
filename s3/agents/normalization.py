import json
import re


def extract_json_block(text: str) -> dict:
    """
    Extract the first valid JSON object from a string.
    Falls back safely if parsing fails.
    """

    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Regex to extract JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        print("No JSON block found.")
        return {
            "decision": "FLAG",
            "issues": []
        }

    try:
        return json.loads(match.group(0))
    except Exception:
        return {
            "decision": "FLAG",
            "issues": []
        }
