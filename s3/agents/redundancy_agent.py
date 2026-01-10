from s3.agents.base import BaseValidationAgent
from s3.agents.normalization import extract_json_block


class RedundancyAgent(BaseValidationAgent):
    """
    Redundancy Validation Agent (S3)

    - Group-scope only
    - Non-gated
    """

    def __init__(self, llm, prompt: str):
        super().__init__(llm)
        self.prompt = prompt

    def run(self, input_data: dict) -> dict:
        group = input_data["group"]

        joined = "\n".join(
            f"- {req['text']}" for req in group
        )

        prompt = self.prompt.replace(
            "{{REQUIREMENT}}", joined
        )

        raw = self.llm.generate(prompt)["text"]
        print("Redundancy Agent Raw Output:", raw)
        result = extract_json_block(raw)

        return {
            "redundancy": {
                "agent": "redundancy",
                "mode": "group",
                "decision": result.get("decision", "FLAG"),
                "issues": result.get("issues", []),
            }
        }
