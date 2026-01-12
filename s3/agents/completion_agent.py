import json
from s3.agents.base import BaseValidationAgent
from s3.agents.normalization import extract_json_block


class CompletionAgent(BaseValidationAgent):
    """
    Completion Validation Agent (S3)

    - Supports single and group modes
    - Non-gated
    """

    def __init__(self, llm, prompts: dict[str, str]):
        super().__init__(llm)
        self.prompts = prompts

    def run(self, input_data: dict) -> dict:
        mode = input_data["mode"]

        if mode == "single":
            return self._run_single(input_data)

        if mode == "group":
            return self._run_group(input_data)

        raise ValueError(f"Unknown completion mode: {mode}")

    # -------------------------------------------------
    # Single-scope completion
    # -------------------------------------------------
    def _run_single(self, input_data: dict) -> dict:
        text = input_data["requirement"]["text"]

        prompt = self.prompts["single"].replace(
            "{{REQUIREMENT}}", text
        )

        raw = self.llm.generate(prompt)["text"]
        result = extract_json_block(raw)

        return {
            "completion_single": {
                "agent": "completion",
                "mode": "single",
                "decision": result["decision"],
                "issues": result.get("issues", []),
            }
        }

    # -------------------------------------------------
    # Group-scope completion
    # -------------------------------------------------
    def _run_group(self, input_data: dict) -> dict:
        group = input_data["group"]

        joined = "\n".join(
            f"- {req['text']}" for req in group
        )
        
        prompt = self.prompts["group"].replace(
            "{{REQUIREMENT}}", joined
        )

        raw = self.llm.generate(prompt)["text"]
        result = extract_json_block(raw)

        return {
            "completion_group": {
                "agent": "completion",
                "mode": "group",
                "decision": result["decision"],
                "issues": result.get("issues", []),
            }
        }