from s3.agents.base import BaseValidationAgent
from s3.agents.normalization import extract_json_block
from s3.agents.robuster import MajorityArbitrator


class ConsistencyAgent(BaseValidationAgent):

    RUNS = 1

    def __init__(self, llm, prompts: dict[str, str]):
        super().__init__(llm)
        self.prompts = prompts

    def run(self, input_data: dict) -> dict:
        mode = input_data["mode"]

        prompt, output_key = self._build_prompt(input_data, mode)
        raw = self.llm.generate(prompt)["text"]
        result = extract_json_block(raw)
        # arbitration = self._execute_redundant(prompt)

        return {
            "consistency_" + mode: {
                "agent": "consistency",
                "decision": result["decision"],
                "issues": result.get("issues", []),
            }
        }

    # -------------------------------------------------
    # Prompt construction only
    # -------------------------------------------------
    def _build_prompt(self, input_data: dict, mode: str) -> tuple[str, str]:

        if mode == "single":
            text = input_data["requirement"]["text"]
            prompt = self.prompts["single"].replace(
                "{{REQUIREMENT}}", text
            )
            return prompt, "consistency_single"

        if mode == "group":
            group = input_data["group"]
            joined = "\n".join(
                f"- {req['text']}" for req in group
            )
            prompt = self.prompts["group"].replace(
                "{{REQUIREMENT}}", joined
            )
            return prompt, "consistency_group"

        raise ValueError(f"Unknown consistency mode: {mode}")

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
