from s3.agents.base import BaseValidationAgent
from utils.normalization import extract_json_block


class CompletionAgent(BaseValidationAgent):
  
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
            output_key: {
                "agent": "completion",
                "decision": result["decision"],
                "issues": result.get("issues", []),
            }
        }

    # -------------------------------------------------
    # Prompt construction only (no logic)
    # -------------------------------------------------
    def _build_prompt(self, input_data: dict, mode: str) -> tuple[str, str]:
        if mode == "single":
            requirement_text = input_data["requirement"].text
            prompt = self.prompts["single"].replace(
                "{{REQUIREMENT}}", requirement_text
            )
            return prompt, "completion_single"

        if mode == "group":
            group = input_data["group"]
            joined = "\n".join(
                f"- {req.text}" for req in group
            )
            prompt = self.prompts["group"].replace(
                "{{REQUIREMENT}}", joined
            )
            return prompt, "completion_group"

        raise ValueError(f"Unknown completion mode: {mode}")

