import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_config() -> dict:
    config = {}
    for name in ("global", "model", "paths"):
        filepath = CONFIG_DIR / f"{name}.yaml"
        if filepath.exists():
            with open(filepath, encoding="utf-8") as f:
                config[name] = yaml.safe_load(f)
    return config
