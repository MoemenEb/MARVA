# s3/agents/__init__.py

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


def build_agents():
    # -------------------------------------------------
    # CachedOllamaClient instances for all agents with system prompt caching
    # System prompt sent ONCE during initialization, cached for all calls
    # -------------------------------------------------
    cfg = load_config()
    atomicity_llm = CachedOllamaClient(
        model=cfg["model"]["model_name"],
        base_url=cfg["model"]["host"],
        system_prompt=load_prompt("atomicity", category="s3/system_prompts"),
        temperature=cfg["model"]["temperature"],
        timeout=cfg["global"]["timeout_seconds"],
        max_retries=cfg["global"]["max_retries"],
    )

    clarity_llm = CachedOllamaClient(
        model=cfg["model"]["model_name"],
        base_url=cfg["model"]["host"],
        system_prompt=load_prompt("clarity", category="s3/system_prompts"),
        temperature=cfg["model"]["temperature"],
        timeout=cfg["global"]["timeout_seconds"],
        max_retries=cfg["global"]["max_retries"],
    )

    redundancy_llm = CachedOllamaClient(
        model=cfg["model"]["model_name"],
        base_url=cfg["model"]["host"],
        system_prompt=load_prompt("redundancy", category="s3/system_prompts"),
        temperature=cfg["model"]["temperature"],
        timeout=cfg["global"]["timeout_seconds"],
        max_retries=cfg["global"]["max_retries"],
    )

    consistency_llm = CachedOllamaClient(
        model=cfg["model"]["model_name"],
        base_url=cfg["model"]["host"],
        system_prompt=load_prompt("consistency", category="s3/system_prompts"),
        temperature=cfg["model"]["temperature"],
        timeout=cfg["global"]["timeout_seconds"],
        max_retries=cfg["global"]["max_retries"],
    )

    completion_single_llm = CachedOllamaClient(
        model=cfg["model"]["model_name"],
        base_url=cfg["model"]["host"],
        system_prompt=load_prompt("completion_single", category="s3/system_prompts"),
        temperature=cfg["model"]["temperature"],
        timeout=cfg["global"]["timeout_seconds"],
        max_retries=cfg["global"]["max_retries"],
    )

    completion_group_llm = CachedOllamaClient(
        model=cfg["model"]["model_name"],
        base_url=cfg["model"]["host"],
        system_prompt=load_prompt("completion_group", category="s3/system_prompts"),
        temperature=cfg["model"]["temperature"],
        timeout=cfg["global"]["timeout_seconds"],
        max_retries=cfg["global"]["max_retries"],
    )

    decision_llm = CachedOllamaClient(
        model=cfg["model"]["model_name"],
        base_url=cfg["model"]["host"],
        system_prompt=load_prompt("decision", category="s3/system_prompts"),
        temperature=cfg["model"]["temperature"],
        timeout=cfg["global"]["timeout_seconds"],
        max_retries=cfg["global"]["max_retries"],
    )

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
    return {
        "atomicity": AtomicityAgent(llm=atomicity_llm, prompts=atomicity_prompts),
        "clarity": ClarityAgent(llm=clarity_llm, prompts=clarity_prompts),
        "redundancy": RedundancyAgent(llm=redundancy_llm, prompts=redundancy_prompts),
        "consistency_single": ConsistencyAgent(llm=consistency_llm, prompts=consistency_prompts),
        "consistency_group": ConsistencyAgent(llm=consistency_llm, prompts=consistency_prompts),
        "completion_single": CompletionAgent(llm=completion_single_llm, prompts=completion_prompts),
        "completion_group": CompletionAgent(llm=completion_group_llm, prompts=completion_prompts),
        "decision": DecisionAgent(llm=decision_llm, prompts=decision_prompts),
    }
