from common.llm_client import LLMClient
from common.prompt_loader import load_prompt


class S1Pipeline:

    def __init__(
        self,
        llm: LLMClient,
        single_prompt_name: str = "s1/s1_single",
        group_prompt_name: str = "s1/s1_group",
    ):
        self.llm = llm
        self.single_prompt = load_prompt(single_prompt_name)
        self.group_prompt = load_prompt(group_prompt_name)

    def run_single(self, requirement: dict) -> dict:
        prompt = self.single_prompt.replace(
            "{{REQUIREMENT}}", requirement["text"]
        )

        response = self.llm.generate(prompt)

        return {
            "mode": "single",
            "req_id": requirement["req_id"],
            "source": requirement["source"],
            "llm_output": response["text"],
            "latency_ms": response["latency_ms"],
        }

    def run_group(self, requirements: list[dict]) -> dict:
        joined_reqs = "\n".join(
            f"[{r['req_id']}] {r['text']}" for r in requirements
        )

        prompt = self.group_prompt.replace(
            "{{REQUIREMENT}}", joined_reqs
        )

        response = self.llm.generate(prompt)

        return {
            "mode": "group",
            "req_ids": [r["req_id"] for r in requirements],
            "sources": list({r["source"] for r in requirements}),
            "llm_output": response["text"],
            "latency_ms": response["latency_ms"],
        }
