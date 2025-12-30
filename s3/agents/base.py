class BaseValidationAgent:
    """
    Base class for all S3 validation agents.

    This class defines what makes something an *agent*
    in MARVA S3:

    - Role identity
    - Local reasoning responsibility
    - Structured output
    """

    def __init__(self, llm):
        self.llm = llm
        self.role = self.__class__.__name__

    def run(self, input_data: dict) -> dict:
        """
        Each agent must implement its own reasoning logic.

        input_data may include:
        - requirement text
        - group context
        - previous agent outputs (later)

        Returns a structured belief, not raw text.
        """
        raise NotImplementedError
