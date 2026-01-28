# s3/agents/__init__.py

from common.llm_client import LLMClient
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
    def create_agent_llm():
        return LLMClient(
            host="http://localhost:11434",
            model="qwen3:1.7b",
            temperature=0.0,
        )

    # -------------------------------------------------
    # Load prompts
    # -------------------------------------------------
    atomicity_prompts = {
        "initial": open("./prompts/atomicity.txt").read(),
        "reflection": open("./prompts/s3_reflections/ref_atomic.txt").read(),
    }
    clarity_prompt = open("./prompts/clarity.txt").read()

    completion_prompts = {
    "single": open("./prompts/completion_single.txt").read(),
    "group": open("./prompts/completion_group.txt").read(),
    }

    consistency_prompts = {
    "single": open("./prompts/consistency_single.txt").read(),
    "group": open("./prompts/consistency_group.txt").read(),
    }

    redundancy_prompt = open("./prompts/redundancy.txt").read()

    vda_prompt = open("./prompts/recommend.txt").read()


    return {
        "atomicity": AtomicityAgent(llm=create_agent_llm(), prompts=atomicity_prompts),
        "clarity": ClarityAgent(llm=create_agent_llm(), prompt=clarity_prompt),
        "completion_single": CompletionAgent(llm=create_agent_llm(), prompts=completion_prompts),
        "completion_group": CompletionAgent(llm=create_agent_llm(), prompts=completion_prompts),
        "consistency_single": ConsistencyAgent(llm=create_agent_llm(), prompts=consistency_prompts),
        "consistency_group": ConsistencyAgent(llm=create_agent_llm(), prompts=consistency_prompts),
        "redundancy": RedundancyAgent(llm=create_agent_llm(), prompt=redundancy_prompt),
        "decision": DecisionAgent(llm=create_agent_llm(), prompt=vda_prompt),

    }
