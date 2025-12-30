class ValidationDecisionAgent:
    """
    Final decision authority in MARVA S3.

    This agent:
    - Receives outputs from all validation agents
    - Reasons over agreement / conflict
    - Produces the final validation decision
    """

    def __init__(self, llm):
        self.llm = llm
        self.role = "ValidationDecisionAgent"

    def run(self, agent_outputs: list[dict]) -> dict:
        summary = "\n".join(
            f"- {o['dimension']}: {o['judgment']}"
            for o in agent_outputs
        )

        prompt = f"""
        You are the final validation authority.

        Given the following validation judgments:
        {summary}

        Identify:
        - Major issues
        - Conflicts between agents
        - Overall validation status
        """

        decision = self.llm.generate(prompt)["text"]

        return {
            "agent": self.role,
            "final_decision": decision
        }
