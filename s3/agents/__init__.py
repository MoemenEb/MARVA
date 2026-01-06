# s3/agents/__init__.py

from common.llm_client import LLMClient
from s3.agents.atomicity_agent import AtomicityAgent
from s3.agents.clarity_agent import ClarityAgent
from s3.agents.stubs import StubAgent, DecisionStubAgent


def build_stub_agents():
    # -------------------------------------------------
    # Single shared LLM for all agents
    # -------------------------------------------------
    llm = LLMClient(
        host="http://localhost:11434",
        model="llama3.2",
        temperature=0.0,
        )

    # -------------------------------------------------
    # Load prompts
    # -------------------------------------------------
    atomicity_prompts = {
        "initial": open("./prompts/atomicity.txt").read(),
        "reflection": open("./prompts/atomicity.txt").read(),
    }
    clarity_prompt = open("./prompts/clarity.txt").read()

    return {
        # REAL agent
        "atomicity": AtomicityAgent(llm=llm, prompts=atomicity_prompts),
        "clarity": ClarityAgent(llm=llm, prompt=clarity_prompt),
        # STUB agents (unchanged)
        "completion_single": StubAgent("completion_single"),
        "consistency_single": StubAgent("consistency_single"),
        "redundancy": StubAgent("redundancy"),
        "completion_group": StubAgent("completion_group"),
        "consistency_group": StubAgent("consistency_group"),
        "decision": DecisionStubAgent(),
    }
