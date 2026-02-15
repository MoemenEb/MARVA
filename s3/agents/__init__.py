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

# Map LLM client names to agents config keys (where they differ)
_CLIENT_TO_CONFIG = {
    "consistency": "consistency_group",
}


def _build_mode_agents(mode, llm_clients, task_prompt):
    """Build only the agent instances required for the given mode."""
    decision_prompts = {"task": load_prompt("decision_task", category="s3/task_prompts")}
    shared = {"task": task_prompt}

    if mode == "single":
        agents = {}
        if "atomicity" in llm_clients:
            agents["atomicity"] = AtomicityAgent(llm=llm_clients["atomicity"], prompts=shared)
        if "clarity" in llm_clients:
            agents["clarity"] = ClarityAgent(llm=llm_clients["clarity"], prompts=shared)
        if "completion_single" in llm_clients:
            agents["completion_single"] = CompletionAgent(llm=llm_clients["completion_single"], prompts=shared)
        active_validators = [k for k in agents]
        agents["decision"] = DecisionAgent(llm=llm_clients["decision"], prompts=decision_prompts, active_validators=active_validators)
        return agents

    agents = {}
    if "redundancy" in llm_clients:
        agents["redundancy"] = RedundancyAgent(llm=llm_clients["redundancy"], prompts=shared)
    if "consistency" in llm_clients:
        agents["consistency_group"] = ConsistencyAgent(llm=llm_clients["consistency"], prompts=shared)
    if "completion_group" in llm_clients:
        agents["completion_group"] = CompletionAgent(llm=llm_clients["completion_group"], prompts=shared)
    active_validators = [k for k in agents]
    agents["decision"] = DecisionAgent(llm=llm_clients["decision"], prompts=decision_prompts, active_validators=active_validators)
    return agents


def build_agents(mode: str):
    overall_start = time.perf_counter()
    logger.info("Building S3 agents for mode='%s'", mode)

    cfg = load_config()
    agents_config = cfg.get("agents", {})

    # -------------------------------------------------
    # Only init LLM clients needed for this mode
    # Filter out disabled agents (decision is always kept)
    # -------------------------------------------------
    all_client_names = _MODE_LLM_CLIENTS[mode]
    client_names = [
        name for name in all_client_names
        if name == "decision" or agents_config.get(
            _CLIENT_TO_CONFIG.get(name, name), {}
        ).get("enabled", True)
    ]
    disabled = set(all_client_names) - set(client_names)
    if disabled:
        logger.info("Disabled agents (skipping LLM init): %s", disabled)

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
    return agents, agents_config
