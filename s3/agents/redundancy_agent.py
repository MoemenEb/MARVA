from s3.agents.base import BaseValidationAgent
from s3.agents.normalization import extract_json_block
from s3.agents.robuster import MajorityArbitrator


class RedundancyAgent(BaseValidationAgent):
    """
    Redundancy Validation Agent (S3)

    - Group-scope only
    - Non-gated
    """

    RUNS = 3

    def __init__(self, llm, prompt: str):
        super().__init__(llm)
        self.prompt = prompt

    def run(self, input_data: dict) -> dict:
        group = input_data["group"]

        prompt = self._build_prompt(group)
        arbitration = self._execute_redundant(prompt)

        return {
            "redundancy": {
                "agent": "redundancy",
                "mode": "group",
                "decision": arbitration["final_decision"],
                "confidence": arbitration["confidence"],
                "issues": arbitration["issues"],
                "raw_runs": arbitration["runs"],
            }
        }

    # -------------------------------------------------
    # Prompt construction
    # -------------------------------------------------
    def _build_prompt(self, group: list[dict]) -> str:
        joined = "\n".join(
            f"- {req['text']}" for req in group
        )

        return self.prompt.replace(
            "{{REQUIREMENT}}", joined
        )

    # -------------------------------------------------
    # Redundant execution + arbitration
    # -------------------------------------------------
    def _execute_redundant(self, prompt: str) -> dict:
        runs = []

        for _ in range(self.RUNS):
            raw = self.llm.generate(prompt)["text"]
            result = extract_json_block(raw)

            runs.append({
                "decision": result.get("decision", "FLAG"),
                "issues": result.get("issues", []),
            })

        return MajorityArbitrator.arbitrate(runs)
