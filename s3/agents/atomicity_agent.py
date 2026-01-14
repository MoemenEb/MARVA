from s3.agents.base import BaseValidationAgent
from s3.agents.normalization import extract_json_block
from s3.agents.robuster import MajorityArbitrator

class AtomicityAgent(BaseValidationAgent):
    RUNS = 3

    def __init__(self, llm, prompts: dict[str, str]):
        super().__init__(llm)
        self.prompts = prompts

    def single_run(self, requirement_text: dict) -> dict:

        # -------------------------------------------------
        # Step 1 — Initial judgment
        # -------------------------------------------------
        initial_prompt = self.prompts["initial"].replace(
            "{{REQUIREMENT}}", requirement_text
        )

        initial_raw = self.llm.generate(initial_prompt)["text"]

        # -------------------------------------------------
        # Step 2 — Normalize output
        # -------------------------------------------------
        result = extract_json_block(initial_raw)

        return {
                "decision": result["decision"],
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
            "atomicity": {
                "agent": "atomicity",
                "mode": "redundant",
                "runs": self.RUNS,
                "decision": arbitration["final_decision"],
                "confidence": arbitration["confidence"],
                "issues": arbitration["issues"],
                "raw_runs": arbitration["runs"],
            }
        }