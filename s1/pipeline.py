from common.llm_client import LLMClient
from common.prompt_loader import load_prompt

class S1Pipeline:
    """
    S1: Single-call, monolithic baseline.
    Exactly one LLM call per requirement.
    """

    def __init__(self, llm: LLMClient, prompt_name: str = "s1_combined"):
        self.llm = llm
        self.prompt_template = load_prompt(prompt_name)

    def run(self, requirement: dict) -> dict:
        prompt = self.prompt_template.replace(
            "{{REQUIREMENT}}", requirement["text"]
        )
        response = self.llm.generate(prompt)

        return {
            "req_id": requirement["req_id"],
            "source": requirement["source"],
            "llm_output": response["text"],
            "latency_ms": response["latency_ms"],
            "model": response["model"]
        }
