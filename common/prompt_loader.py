from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = PROJECT_ROOT / "prompts"

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
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")
