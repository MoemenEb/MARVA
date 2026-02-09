# s3/agents/__init__.py

import logging
import time
from common.cached_ollama_client import CachedOllamaClient
from common.config import load_config
from common.prompt_loader import load_prompt
from s3.agents.atomicity_agent import AtomicityAgent
from s3.agents.clarity_agent import ClarityAgent
from s3.agents.completion_agent import CompletionAgent
from s3.agents.consistency_agent import ConsistencyAgent
from s3.agents.decision_agent import DecisionAgent
from s3.agents.redundancy_agent import RedundancyAgent

logger = logging.getLogger("marva.s3.agents")


# LLM client names needed per mode (decision is always included)
_MODE_LLM_CLIENTS = {
    "single": ["atomicity", "clarity", "completion_single", "decision"],
    "group":  ["redundancy", "consistency", "completion_group", "decision"],
}


def _build_mode_agents(mode, llm_clients, task_prompt):
    """Build only the agent instances required for the given mode."""
    decision_prompts = {"task": load_prompt("decision_task", category="s3/task_prompts")}
    shared = {"task": task_prompt}

    if mode == "single":
        return {
            "atomicity": AtomicityAgent(llm=llm_clients["atomicity"], prompts=shared),
            "clarity": ClarityAgent(llm=llm_clients["clarity"], prompts=shared),
            "completion_single": CompletionAgent(llm=llm_clients["completion_single"], prompts=shared),
            "decision": DecisionAgent(llm=llm_clients["decision"], prompts=decision_prompts),
        }

    return {
        "redundancy": RedundancyAgent(llm=llm_clients["redundancy"], prompts=shared),
        "consistency_group": ConsistencyAgent(llm=llm_clients["consistency"], prompts=shared),
        "completion_group": CompletionAgent(llm=llm_clients["completion_group"], prompts=shared),
        "decision": DecisionAgent(llm=llm_clients["decision"], prompts=decision_prompts),
    }


def build_agents(mode: str):
    overall_start = time.perf_counter()
    logger.info("Building S3 agents for mode='%s'", mode)

    cfg = load_config()

    # -------------------------------------------------
    # Only init LLM clients needed for this mode
    # -------------------------------------------------
    client_names = _MODE_LLM_CLIENTS[mode]
    llm_clients = {}

    for name in client_names:
        t0 = time.perf_counter()
        logger.info("Initializing cached LLM client for '%s'", name)
        llm_clients[name] = CachedOllamaClient(
            model=cfg["model"]["model_name"],
            base_url=cfg["model"]["host"],
            system_prompt=load_prompt(name, category="s3/system_prompts"),
            temperature=cfg["model"]["temperature"],
            num_predict=cfg["model"].get("max_tokens", 1024),
            timeout=cfg["global"]["timeout_seconds"],
            max_retries=cfg["global"]["max_retries"],
        )
        logger.info("Cached LLM client '%s' ready in %.2fs", name, time.perf_counter() - t0)

    # -------------------------------------------------
    # Build mode-specific agents
    # -------------------------------------------------
    shared_task_prompt = load_prompt("shared_task", category="s3/task_prompts")
    agents = _build_mode_agents(mode, llm_clients, shared_task_prompt)

    overall_elapsed = time.perf_counter() - overall_start
    logger.info("Built %d agents for mode='%s' in %.2fs", len(agents), mode, overall_elapsed)
    return agents
