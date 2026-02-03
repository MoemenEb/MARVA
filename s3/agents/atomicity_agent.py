from s3.agents.base import BaseValidationAgent
from utils.normalization import extract_json_block
from entity.agent import AgentResult

class AtomicityAgent(BaseValidationAgent):

    def __init__(self, llm, prompts: dict[str, str]):
        super().__init__(llm)
        self.prompts = prompts

    def run(self, input_data: dict) -> dict:
        requirement_text = input_data["requirement"].text
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

        agent = AgentResult(
            agent="atomicity",
            status=result.get("decision", "FLAG"),
            issues=result.get("issues", [])
        )
 
        return {"atomicity": agent}