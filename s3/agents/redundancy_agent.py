from s3.agents.base import BaseValidationAgent
from s3.agents.normalization import extract_json_block


class RedundancyAgent(BaseValidationAgent):
    def __init__(self, llm, prompt: str):
        super().__init__(llm)
        self.prompt = prompt

    def run(self, input_data: dict) -> dict:
        group = input_data["group"]

        prompt = self._build_prompt(group)
        raw = self.llm.generate(prompt)["text"]
        result = extract_json_block(raw)

        return {
            "redundancy": {
                "agent": "redundancy",
                "mode": "group",
                "decision": result["decision"],
                "issues": result.get("issues", []),
            }
        }

    # -------------------------------------------------
    # Prompt construction
    # -------------------------------------------------
    def _build_prompt(self, group: list[dict]) -> str:
        joined = "\n".join(
            f"- {req.text}" for req in group
        )

        return self.prompt.replace(
            "{{REQUIREMENT}}", joined
        )
