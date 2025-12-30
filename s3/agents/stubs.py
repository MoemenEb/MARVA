class StubAgent:
    def __init__(self, name, status="pass"):
        self.name = name
        self.status = status

    def __call__(self, state: dict) -> dict:
        # IMPORTANT:
        # Return ONLY the key this agent owns
        return {
            self.name: {
                "agent": self.name,
                "status": self.status,
                "message": f"{self.name} executed"
            }
        }

class DecisionStubAgent:
    """
    Stub for the final decision agent.

    IMPORTANT:
    - Must also write to a unique key ('decision')
    - Must NOT return full state
    """

    def __call__(self, state: dict) -> dict:
        return {
            "decision": {
                "agent": "decision",
                "summary": "decision executed",
                "inputs_seen": list(state.keys())
            }
        }