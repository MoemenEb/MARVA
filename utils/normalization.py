import json
import logging
import re

logger = logging.getLogger(__name__)


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
    match = re.search(r"(?:```json\s*)?(\{.*?\})(?:\s*```)?", text, re.DOTALL)

    if not match:
        logger.warning("No JSON block found in LLM response.")
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
