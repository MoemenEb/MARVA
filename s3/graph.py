from langgraph.graph import StateGraph, START, END
from s3.state import MARVAState


# ------------------------------------------------------------------
# Control-only nodes (no state writes)
# ------------------------------------------------------------------

def orchestrator_agent(state: MARVAState):
    """
    Master MARVA Orchestrator Agent.

    Conceptual role:
    - Controls validation flow
    - Can later be extended with reasoning / LLM logic
    - Does NOT write validation results
    """
    return None


def single_parallel_node(state: MARVAState):
    """Fan-out node for single-requirement validation."""
    return None


def group_parallel_node(state: MARVAState):
    """Fan-out node for group-level validation."""
    return None


def join_node(state: MARVAState):
    """Join / synchronization node before decision."""
    return None


# ------------------------------------------------------------------
# Graph builder
# ------------------------------------------------------------------

def build_marva_s3_graph(agents: dict):
    """
    MARVA S3 Orchestration Graph (FINAL)

    Properties:
    - Explicit Master Orchestrator agent
    - Atomicity hard gate (single mode only)
    - Parallel validation where required
    - Join nodes enforce fan-in
    - Typed state (MARVAState) prevents root conflicts
    """

    graph = StateGraph(MARVAState)

    # -------------------------------------------------
    # Nodes
    # -------------------------------------------------

    # Master orchestrator
    graph.add_node("orchestrator", orchestrator_agent)

    # Single-scope validation agents
    graph.add_node("atomicity", agents["atomicity"])
    graph.add_node("clarity", agents["clarity"])
    graph.add_node("completion_single", agents["completion_single"])
    graph.add_node("consistency_single", agents["consistency_single"])

    # Group-scope validation agents
    graph.add_node("redundancy", agents["redundancy"])
    graph.add_node("completion_group", agents["completion_group"])
    graph.add_node("consistency_group", agents["consistency_group"])

    # Control / synchronization nodes
    graph.add_node("single_parallel", single_parallel_node)
    graph.add_node("group_parallel", group_parallel_node)
    graph.add_node("join_single", join_node)
    graph.add_node("join_group", join_node)

    # Decision agent
    graph.add_node("decision", agents["decision"])

    # -------------------------------------------------
    # Entry point
    # -------------------------------------------------

    graph.set_entry_point("orchestrator")

    # -------------------------------------------------
    # Orchestrator routing (flow control)
    # -------------------------------------------------

    def orchestrator_router(state: MARVAState):
        if state["mode"] == "single":
            return "atomicity"
        if state["mode"] == "group":
            return "group_parallel"
        raise ValueError(f"Unknown mode: {state['mode']}")

    graph.add_conditional_edges(
        "orchestrator",
        orchestrator_router,
        {
            "atomicity": "atomicity",
            "group_parallel": "group_parallel",
        },
    )

    # -------------------------------------------------
    # Atomicity hard gate (single mode only)
    # -------------------------------------------------

    def atomicity_router(state: MARVAState):
        if state["atomicity"]["status"] == "fail":
            return "decision"
        return "single_parallel"

    graph.add_conditional_edges(
        "atomicity",
        atomicity_router,
        {
            "decision": "decision",
            "single_parallel": "single_parallel",
        },
    )

    # -------------------------------------------------
    # Single-scope parallel validation
    # -------------------------------------------------

    graph.add_edge("single_parallel", "clarity")
    graph.add_edge("single_parallel", "completion_single")
    graph.add_edge("single_parallel", "consistency_single")

    graph.add_edge("clarity", "join_single")
    graph.add_edge("completion_single", "join_single")
    graph.add_edge("consistency_single", "join_single")

    graph.add_edge("join_single", "decision")

    # -------------------------------------------------
    # Group-scope parallel validation
    # -------------------------------------------------

    graph.add_edge("group_parallel", "redundancy")
    graph.add_edge("group_parallel", "completion_group")
    graph.add_edge("group_parallel", "consistency_group")

    graph.add_edge("redundancy", "join_group")
    graph.add_edge("completion_group", "join_group")
    graph.add_edge("consistency_group", "join_group")

    graph.add_edge("join_group", "decision")

    # -------------------------------------------------
    # End
    # -------------------------------------------------

    graph.add_edge("decision", END)

    return graph
