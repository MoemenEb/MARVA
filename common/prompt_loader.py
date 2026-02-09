import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = PROJECT_ROOT / "prompts"
logger = logging.getLogger("marva.prompt_loader")

def load_prompt(name: str, category: str = None) -> str:
    """
    Load prompt from prompts directory.

    Args:
        name: Prompt filename without .txt extension
        category: Optional subdirectory path (e.g., "s3/system_prompts")

    Returns:
        Prompt text content
    """
    if category:
        path = PROMPT_DIR / category / f"{name}.txt"
    else:
        path = PROMPT_DIR / f"{name}.txt"

    if not path.exists():
        logger.error("Prompt not found: %s", path)
        raise FileNotFoundError(f"Prompt not found: {path}")
    content = path.read_text(encoding="utf-8")
    logger.debug("Loaded prompt '%s' (%d chars)", name, len(content))
    return content
