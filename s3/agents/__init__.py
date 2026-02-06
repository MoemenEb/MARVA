# s3/agents/__init__.py

from common.llm_client import LLMClient
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
    # Single shared LLM for all agents
    # -------------------------------------------------
    cfg = load_config()
    def create_agent_llm():
        return LLMClient(
            host=cfg["model"]["host"],
            model=cfg["model"]["model_name"],
            temperature=cfg["model"]["temperature"],
            timeout=cfg["global"]["timeout_seconds"],
            max_retries=cfg["global"]["max_retries"],
        )

    # -------------------------------------------------
    # Load prompts
    # -------------------------------------------------
    atomicity_prompts = {
        "initial": load_prompt("atomicity"),
    }

    completion_prompts = {
        "single": load_prompt("completion_single"),
        "group": load_prompt("completion_group"),
    }

    consistency_prompts = {
        "single": load_prompt("consistency_single"),
        "group": load_prompt("consistency_group"),
    }

    return {
        "atomicity": AtomicityAgent(llm=create_agent_llm(), prompts=atomicity_prompts),
        "clarity": ClarityAgent(llm=create_agent_llm(), prompt=load_prompt("clarity")),
        "completion_single": CompletionAgent(llm=create_agent_llm(), prompts=completion_prompts),
        "completion_group": CompletionAgent(llm=create_agent_llm(), prompts=completion_prompts),
        "consistency_single": ConsistencyAgent(llm=create_agent_llm(), prompts=consistency_prompts),
        "consistency_group": ConsistencyAgent(llm=create_agent_llm(), prompts=consistency_prompts),
        "redundancy": RedundancyAgent(llm=create_agent_llm(), prompt=load_prompt("redundancy")),
        "decision": DecisionAgent(llm=create_agent_llm(), prompt=load_prompt("recommend")),
    }
