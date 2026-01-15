from s3.agents.base import BaseValidationAgent
from s3.agents.normalization import extract_json_block
from s3.agents.robuster import MajorityArbitrator

class ClarityAgent(BaseValidationAgent):
    RUNS = 1
  
    def __init__(self, llm, prompt: str):
        super().__init__(llm)
        self.prompt = prompt

    def single_run(self, requirement: str) -> dict:
        filled_prompt = self.prompt.replace(
            "{{REQUIREMENT}}", requirement
        )

        response = self.llm.generate(filled_prompt)["text"]

        result = extract_json_block(response)

        return {
                "decision": result.get("decision", "FLAG"),
                "issues": result.get("issues", []),
        }
    
    def run(self, input_data: dict) -> dict:

        requirement_text = input_data["requirement"]["text"]

        # -------------------------------------------------
        # Step 1 — Redundant executions
        # -------------------------------------------------
        runs = [
            self.single_run(requirement_text)
            for _ in range(self.RUNS)
        ]

        # -------------------------------------------------
        # Step 2 — Majority arbitration
        # -------------------------------------------------
        arbitration = MajorityArbitrator.arbitrate(runs)

        # -------------------------------------------------
        # Step 3 — Final normalized output
        # -------------------------------------------------
        return {
            "clarity": {
                "agent": "clarity",
                "mode": "redundant",
                "runs": self.RUNS,
                "decision": arbitration["final_decision"],
                "confidence": arbitration["confidence"],
                "issues": arbitration["issues"],
                "raw_runs": arbitration["runs"],
            }
        }
