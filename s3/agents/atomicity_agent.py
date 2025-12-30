from s3.agents.base import BaseValidationAgent


class AtomicityAgent(BaseValidationAgent):
    """
    Atomicity Agent (S3)

    This agent:
    - Reasons only about atomicity
    - Forms an initial judgment
    - Reflects and revises
    """

    def run(self, input_data: dict) -> dict:
        requirement = input_data["requirement"]

        # Step 1 — Initial judgment
        initial_prompt = f"""
        Determine whether the following requirement is atomic.
        If not, explain why.

        Requirement:
        {requirement}
        """

        initial = self.llm.generate(initial_prompt)["text"]

        # Step 2 — Reflection
        reflection_prompt = f"""
        Review your previous judgment:

        {initial}

        Are there hidden compound actions or implicit conjunctions?
        Revise your judgment if necessary.
        """

        refined = self.llm.generate(reflection_prompt)["text"]

        return {
            "agent": self.role,
            "dimension": "atomicity",
            "judgment": refined,
            "confidence": "high"
        }
