from s3.agents.base import BaseValidationAgent


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

        # NOTE: For now, we assume the model responds clearly.
        # You can later harden parsing if needed.
        decision = "FLAG" if "FLAG" in response.upper() else "PASS"

        issues = []
        if decision == "FLAG":
            issues.append(response.strip())

        return {
            "clarity": {
                "agent": "clarity",
                "mode": "single",
                "decision": decision,
                "issues": issues
            }
        }
