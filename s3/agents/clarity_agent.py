from s3.agents.base import BaseValidationAgent
from s3.agents.normalization import extract_json_block


class ClarityAgent(BaseValidationAgent):
    """
    Clarity Validation Agent (S3)

    - Single-scope only
    - Non-gated
    - Detects ambiguity / lack of clarity
    """

    def __init__(self, llm, prompt: str):
        super().__init__(llm)
        self.prompt = prompt

    def run(self, input_data: dict) -> dict:
        requirement = input_data["requirement"]["text"]

        filled_prompt = self.prompt.replace(
            "{{REQUIREMENT}}", requirement
        )

        response = self.llm.generate(filled_prompt)["text"]

        result = extract_json_block(response)

        return {
            "clarity": {
                "agent": "clarity",
                "mode": "single",
                "decision": result.get("decision", "FLAG"),
                "issues": result.get("issues", []),
            }
        }
