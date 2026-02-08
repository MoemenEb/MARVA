from langgraph.graph import StateGraph, END
from s3.state import MARVAState
from concurrent.futures import ThreadPoolExecutor
import time
import logging


# ------------------------------------------------------------------
# Control-only nodes (no state writes)
# ------------------------------------------------------------------

def orchestrator_agent(state: MARVAState):
    return {
        "mode": state["mode"],
        "requirement": state.get("requirement"),
        "requirement_set": state.get("requirement_set"),
    }


def single_parallel_node(state: MARVAState):
    """Fan-out node for single-requirement validation."""
    return None


def group_parallel_node(state: MARVAState):
    """Fan-out node for group-level validation."""
    return None


def join_node(state: MARVAState):
    """Join / synchronization node before decision."""
    return None


def execute_parallel_agents(state: MARVAState, agent_list: list, max_workers: int = None):
    """
    Execute multiple agents in parallel using ThreadPoolExecutor.

    Args:
        state: Current state to pass to agents
        agent_list: List of (agent, name) tuples to execute
        max_workers: Number of threads (defaults to len(agent_list))

    Returns:
        Merged results from all agents
    """
    logger = logging.getLogger("marva.s3.graph")

    if max_workers is None:
        max_workers = len(agent_list)

    logger.info(f"Starting parallel execution of {len(agent_list)} agents: {[name for _, name in agent_list]}")
    overall_start = time.perf_counter()

    # Wrapper to time individual agent execution
    def timed_agent_run(agent, name):
        start = time.perf_counter()
        logger.info(f"[{name}] Started execution")
        result = agent.run(state)
        elapsed = time.perf_counter() - start
        logger.info(f"[{name}] Completed in {elapsed:.2f}s")
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all agents for parallel execution
        futures = {
            executor.submit(timed_agent_run, agent, name): name
            for agent, name in agent_list
        }

        # Collect results
        merged_results = {}
        for future in futures:
            agent_name = futures[future]
            try:
                result = future.result()
                merged_results.update(result)
            except Exception as e:
                logger.error(f"[{agent_name}] failed: {e}")
                raise

        overall_elapsed = time.perf_counter() - overall_start
        logger.info(f"Parallel execution completed in {overall_elapsed:.2f}s total")

        return merged_results


# ------------------------------------------------------------------
# Graph builder
# ------------------------------------------------------------------

def build_marva_s3_graph(agents: dict):

    graph = StateGraph(MARVAState)

    # -------------------------------------------------
    # Nodes
    # -------------------------------------------------

    # Master orchestrator
    graph.add_node("orchestrator", orchestrator_agent)

    # Single-scope validation agents
    graph.add_node("atomicity", lambda s: agents["atomicity"].run(s))
    # NOTE: clarity and completion_single now run in parallel via single_parallel_exec node
    # graph.add_node("clarity", lambda s: agents["clarity"].run(s))
    # graph.add_node("completion_single", lambda s: agents["completion_single"].run(s))

    # Group-scope validation agents
    # NOTE: redundancy, completion_group, and consistency_group now run in parallel via group_parallel_exec node
    # graph.add_node("redundancy", lambda s: agents["redundancy"].run(s))
    # graph.add_node("completion_group", lambda s: agents["completion_group"].run(s))
    # graph.add_node("consistency_group", lambda s: agents["consistency_group"].run(s))

    # Control / synchronization nodes
    graph.add_node("single_parallel", single_parallel_node)
    graph.add_node("group_parallel", group_parallel_node)
    # NOTE: join nodes replaced by parallel execution nodes
    # graph.add_node("join_single", join_node)
    # graph.add_node("join_group", join_node)

    # Parallel execution nodes
    def single_parallel_execution(state: MARVAState):
        agent_list = [
            (agents["clarity"], "clarity"),
            (agents["completion_single"], "completion_single")
        ]
        return execute_parallel_agents(state, agent_list, max_workers=2)

    def group_parallel_execution(state: MARVAState):
        agent_list = [
            (agents["redundancy"], "redundancy"),
            (agents["completion_group"], "completion_group"),
            (agents["consistency_group"], "consistency_group")
        ]
        return execute_parallel_agents(state, agent_list, max_workers=3)

    graph.add_node("single_parallel_exec", single_parallel_execution)
    graph.add_node("group_parallel_exec", group_parallel_execution)

    # Decision agent
    graph.add_node("decision", lambda s: agents["decision"].run(s))

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
        if state["atomicity"].status.upper() == "FAIL":
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

    # Parallel execution of clarity and completion_single
    graph.add_edge("single_parallel", "single_parallel_exec")
    graph.add_edge("single_parallel_exec", "decision")

    # -------------------------------------------------
    # Group-scope parallel validation
    # -------------------------------------------------

    # Parallel execution of redundancy, completion_group, and consistency_group
    graph.add_edge("group_parallel", "group_parallel_exec")
    graph.add_edge("group_parallel_exec", "decision")

    # -------------------------------------------------
    # End
    # -------------------------------------------------

    graph.add_edge("decision", END)

    return graph
