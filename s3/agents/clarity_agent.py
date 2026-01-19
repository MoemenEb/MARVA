from s3.agents.base import BaseValidationAgent
from s3.agents.normalization import extract_json_block
from s3.agents.robuster import MajorityArbitrator

class ClarityAgent(BaseValidationAgent):
  
    def __init__(self, llm, prompt: str):
        super().__init__(llm)
        self.prompt = prompt

    def run(self, input_data: dict) -> dict:
        requirement_text = input_data["requirement"]["text"]
        filled_prompt = self.prompt.replace(
            "{{REQUIREMENT}}", requirement_text
        )

        response = self.llm.generate(filled_prompt)["text"]

        result = extract_json_block(response)

        return {
            "clarity": {
                "agent": "clarity",
                "decision": result.get("decision", "FLAG"),
                "issues": result.get("issues", []),
            }
        }
    
