import yaml
import logging
import time
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
logger = logging.getLogger("marva.config")


def load_config() -> dict:
    start = time.perf_counter()
    config = {}
    for name in ("global", "model", "paths", "agents"):
        filepath = CONFIG_DIR / f"{name}.yaml"
        if filepath.exists():
            with open(filepath, encoding="utf-8") as f:
                config[name] = yaml.safe_load(f) or {}
            logger.debug("Loaded config: %s", name)
        else:
            logger.warning("Config file not found: %s", filepath)
            config[name] = {}

    required = {
        "global": ["timeout_seconds", "max_retries"],
        "model": ["host", "model_name", "temperature"],
    }
    missing = []
    for section, keys in required.items():
        for key in keys:
            if key not in config.get(section, {}):
                missing.append(f"{section}.{key}")

    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    # Validate agents config structure
    agents_cfg = config.get("agents", {})
    if "decision" in agents_cfg:
        raise ValueError("agents.decision: the decision agent cannot be configured (always enabled)")
    for agent_name, agent_conf in agents_cfg.items():
        if not isinstance(agent_conf, dict):
            raise ValueError(f"agents.{agent_name}: expected a dict, got {type(agent_conf).__name__}")
        for flag in ("enabled", "hard_gate"):
            if flag not in agent_conf:
                raise ValueError(f"agents.{agent_name}.{flag}: missing required key")
            if not isinstance(agent_conf[flag], bool):
                raise ValueError(f"agents.{agent_name}.{flag}: expected bool, got {type(agent_conf[flag]).__name__}")

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.debug("All configs loaded in %.1fms", elapsed_ms)
    return config
