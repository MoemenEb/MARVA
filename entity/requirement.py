class Requirement:
    def __init__(self, req_id: str, text: str):
        self.id = req_id
        self.text = text

        self.metadata = {}

        # single-scope validation signals only
        self.single_validations = {}

        # VDA outputs
        self.final_decision = None
        self.recommendation = []
