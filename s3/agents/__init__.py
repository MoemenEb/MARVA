from s3.agents.stubs import StubAgent, DecisionStubAgent


def build_stub_agents():
    return {
        # single-scope
        "atomicity": StubAgent("atomicity", status="pass"),
        "clarity": StubAgent("clarity"),
        "completion_single": StubAgent("completion_single"),
        "consistency_single": StubAgent("consistency_single"),

        # group-scope
        "redundancy": StubAgent("redundancy"),
        "completion_group": StubAgent("completion_group"),
        "consistency_group": StubAgent("consistency_group"),

        # decision
        "decision": DecisionStubAgent(),
    }
