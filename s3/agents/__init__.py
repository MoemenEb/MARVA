# s3/agents/__init__.py

import logging
import time
from common.llm_client import LLMClient
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


def build_agents():
    overall_start = time.perf_counter()
    logger.info("Building all S3 agents with cached system prompts")

    # -------------------------------------------------
    # CachedOllamaClient instances for all agents with system prompt caching
    # System prompt sent ONCE during initialization, cached for all calls
    # -------------------------------------------------
    cfg = load_config()

    agent_names = [
        "atomicity", "clarity", "redundancy", "consistency",
        "completion_single", "completion_group", "decision"
    ]
    llm_clients = {}

    for name in agent_names:
        t0 = time.perf_counter()
        logger.info("Initializing cached LLM client for '%s'", name)
        llm_clients[name] = CachedOllamaClient(
            model=cfg["model"]["model_name"],
            base_url=cfg["model"]["host"],
            system_prompt=load_prompt(name, category="s3/system_prompts"),
            temperature=cfg["model"]["temperature"],
            timeout=cfg["global"]["timeout_seconds"],
            max_retries=cfg["global"]["max_retries"],
        )
        logger.info("Cached LLM client '%s' ready in %.2fs", name, time.perf_counter() - t0)

    # -------------------------------------------------
    # Load prompts
    # -------------------------------------------------
    # Shared task prompt for validation agents
    shared_task_prompt = load_prompt("shared_task", category="s3/task_prompts")

    atomicity_prompts = {
        "task": shared_task_prompt,
    }

    clarity_prompts = {
        "task": shared_task_prompt,
    }

    redundancy_prompts = {
        "task": shared_task_prompt,
    }

    consistency_prompts = {
        "task": shared_task_prompt,
    }

    completion_prompts = {
        "task": shared_task_prompt,
    }

    decision_prompts = {
        "task": load_prompt("decision_task", category="s3/task_prompts"),
    }

    # All agents now use CachedOllamaClient with system prompt caching
    agents = {
        "atomicity": AtomicityAgent(llm=llm_clients["atomicity"], prompts=atomicity_prompts),
        "clarity": ClarityAgent(llm=llm_clients["clarity"], prompts=clarity_prompts),
        "redundancy": RedundancyAgent(llm=llm_clients["redundancy"], prompts=redundancy_prompts),
        "consistency_single": ConsistencyAgent(llm=llm_clients["consistency"], prompts=consistency_prompts),
        "consistency_group": ConsistencyAgent(llm=llm_clients["consistency"], prompts=consistency_prompts),
        "completion_single": CompletionAgent(llm=llm_clients["completion_single"], prompts=completion_prompts),
        "completion_group": CompletionAgent(llm=llm_clients["completion_group"], prompts=completion_prompts),
        "decision": DecisionAgent(llm=llm_clients["decision"], prompts=decision_prompts),
    }

    overall_elapsed = time.perf_counter() - overall_start
    logger.info("All %d agents built in %.2fs", len(agents), overall_elapsed)
    return agents
