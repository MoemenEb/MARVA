class AgentResult:
    def __init__(self, agent, status, issues=None):
        self.agent = agent
        self.status = status        # PASS | FLAG | FAIL (FAIL only for Atomicity) | TiMEOUT | ERROR.
        self.issues = issues or []

    def to_dict(self):
        return{
            "agent" : self.agent,
            "status" : self.status,
            "issues" : self.issues
        }

    def __repr__(self) -> str:
        return f"AgentResult(agent={self.agent!r}, status={self.status!r}, issues={self.issues!r})"

