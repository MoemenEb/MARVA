class AgentResult:
    def __init__(self, agent, scope, status, issues=None, evidence=None):
        self.agent = agent
        self.scope = scope          # SINGLE | GROUP
        self.status = status        # PASS | FLAG | FAIL (FAIL only for Atomicity) | TiMEOUT | ERROR.
        self.issues = issues or []