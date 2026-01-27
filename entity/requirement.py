class Requirement:
    def __init__(self, req_id: str, text: str):
        self.id = req_id
        self.text = text

        self.metadata = {}

        # single-scope validation signals only
        self.single_validations = []

        # VDA outputs
        self.final_decision = None
        self.recommendation = {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "single_validations": self.single_validations,
            "final_decision": self.final_decision,
            "recommendation": self.recommendation,
        }
