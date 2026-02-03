from s3.agents.base import BaseValidationAgent
from utils.normalization import extract_json_block
from entity.agent import AgentResult


class RedundancyAgent(BaseValidationAgent):
    def __init__(self, llm, prompt: str):
        super().__init__(llm)
        self.prompt = prompt

    def run(self, input_data: dict) -> dict:
        requirement_set = input_data["requirement_set"]

        prompt = self.prompt.replace(
            "{{REQUIREMENT}}", requirement_set.join_requirements()
        )
        raw = self.llm.generate(prompt)["text"]
        result = extract_json_block(raw)

        return {
            "redundancy": AgentResult(
                agent="redundancy",
                status=result.get("decision", "FLAG"),
                issues=result.get("issues", [])
            )
        }
