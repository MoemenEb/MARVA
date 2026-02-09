import yaml
import logging
import time
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
logger = logging.getLogger("marva.config")


def load_config() -> dict:
    start = time.perf_counter()
    config = {}
    for name in ("global", "model", "paths"):
        filepath = CONFIG_DIR / f"{name}.yaml"
        if filepath.exists():
            with open(filepath, encoding="utf-8") as f:
                config[name] = yaml.safe_load(f)
            logger.debug("Loaded config: %s", name)
        else:
            logger.warning("Config file not found: %s", filepath)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.debug("All configs loaded in %.1fms", elapsed_ms)
    return config
