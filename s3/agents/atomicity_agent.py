from s3.agents.base import BaseValidationAgent
from s3.agents.normalization import extract_json_block
from s3.agents.robuster import MajorityArbitrator

class AtomicityAgent(BaseValidationAgent):

    def __init__(self, llm, prompts: dict[str, str]):
        super().__init__(llm)
        self.prompts = prompts

    def run(self, input_data: dict) -> dict:

        requirement_text = input_data["requirement"]["text"]
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
        # -------------------------------------------------
        # Step 3 — Reflection
        # -------------------------------------------------
        # reflection_prompt = self.prompts["reflection"].replace(  
        #     "{{REQUIREMENT}}", requirement_text
        # ).replace(
        #     "{{DECISION}}", result["decision"]
        # ).replace(
        #     "{{REASON}}", result.get("issues", [])
        # )
        # reflection_raw = self.llm.generate(reflection_prompt)["text"]
        # reflection_result = extract_json_block(reflection_raw)
        # print("Reflection result:", reflection_result)

        return {"atomicity": {
                "decision": result["decision"],
                "issues": result.get("issues", []),
            }
        }