import json
from s3.agents.base import BaseValidationAgent


class AtomicityAgent(BaseValidationAgent):
    """
    Atomicity Validation Agent (S3)

    - Single-scope only
    - Hard gate
    - Two-step reasoning (initial + reflection)
    - Canonical MARVA output
    """

    def __init__(self, llm, prompts: dict[str, str]):
        super().__init__(llm)
        self.prompts = prompts

    def run(self, input_data: dict) -> dict:
        """
        Expects input_data to contain:
        {
            "requirement": {
                "req_id": "...",
                "text": "..."
            }
        }
        """
        requirement_text = input_data["requirement"]["text"]

        # -------------------------------------------------
        # Step 1 — Initial judgment
        # -------------------------------------------------
        initial_prompt = self.prompts["initial"].replace(
            "{{REQUIREMENT}}", requirement_text
        )

        initial_raw = self.llm.generate(initial_prompt)["text"]

        # -------------------------------------------------
        # Step 2 — Reflection
        # -------------------------------------------------
        # reflection_prompt = self.prompts["reflection"].replace(
        #     "{{INITIAL_JUDGMENT}}", initial_raw
        # )

        # refined_raw = self.llm.generate(reflection_prompt)["text"]

        # -------------------------------------------------
        # Step 3 — Normalize output
        # -------------------------------------------------
        result = self._normalize(initial_raw)

        return {
            "atomicity": {
                "agent": "atomicity",
                "mode": "single",
                "decision": result["decision"],
                "issues": result.get("issues", []),
            }
        }

    def _normalize(self, raw: str) -> dict:
        """
        Parse and normalize LLM output into canonical MARVA format.
        """
        try:
            parsed = json.loads(raw)
        except Exception:
            # Hard-gate safety fallback
            return {
                "decision": "FLAG",
                "issues": []
            }

        return {
            "decision": parsed.get("decision", "FLAG"),
            "issues": parsed.get("issues", []),
        }
