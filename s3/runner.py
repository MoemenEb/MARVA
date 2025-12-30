from s3.graph import build_marva_s3_graph
from s3.agents import build_stub_agents


def run_s3(mode: str):
    """
    Minimal S3 execution runner.

    mode: 'single' or 'group'
    """

    # -------------------------
    # Build agents + graph
    # -------------------------
    agents = build_stub_agents()
    graph = build_marva_s3_graph(agents)

    app = graph.compile()

    # -------------------------
    # Initial state
    # -------------------------
    initial_state = {
        "mode": mode,
        "requirement": "The system shall authenticate users using email and password.",
        "group": [
            "The system shall authenticate users.",
            "The system shall log authentication attempts."
        ]
    }

    # -------------------------
    # Execute
    # -------------------------
    final_state = app.invoke(initial_state)

    return final_state


if __name__ == "__main__":
    print("=== SINGLE MODE ===")
    single_result = run_s3("single")
    print(single_result)

    print("\n=== GROUP MODE ===")
    group_result = run_s3("group")
    print(group_result)
