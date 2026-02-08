from s3.agents.base import BaseValidationAgent
from utils.normalization import extract_json_block
from entity.agent import AgentResult

class ConsistencyAgent(BaseValidationAgent):
    def __init__(self, llm, prompts: dict[str, str]):
        super().__init__(llm)
        self.prompts = prompts

    def run(self, input_data: dict) -> dict:
        mode = input_data["mode"]

        prompt, output_key = self._build_prompt(input_data, mode)
        response = self.llm.generate(prompt)
        if response["execution_status"] != "SUCCESS":
            return {output_key: AgentResult(agent=output_key, status="FLAG", issues=[])}
        result = extract_json_block(response["text"])

        return {
            output_key: AgentResult(
                agent=output_key,
                status=result.get("decision", "FLAG"),
                issues=result.get("issues", [])
            )
        }

    # -------------------------------------------------
    # Prompt construction only
    # -------------------------------------------------
    def _build_prompt(self, input_data: dict, mode: str) -> tuple[str, str]:

        # if mode == "single":
        #     requirement_text = input_data["requirement"].text
        #     prompt = self.prompts["single"].replace(
        #         "{{REQUIREMENT}}", requirement_text
        #     )
        #     return prompt, "consistency_single"

        if mode == "group":
            requirement_set = input_data["requirement_set"]
            prompt = self.prompts["group"].replace(
                "{{REQUIREMENT}}", requirement_set.join_requirements()
            )
            return prompt, "consistency_group"

        raise ValueError(f"Unknown consistency mode: {mode}")
