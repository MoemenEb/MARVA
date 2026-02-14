from langgraph.graph import StateGraph, END
from s3.state import MARVAState
from concurrent.futures import ThreadPoolExecutor
import time
import logging


logger = logging.getLogger("marva.s3.graph")


def _get_agent(agents: dict, key: str):
    """Get agent by key with a clear error for uninitialized agents."""
    agent = agents.get(key)
    if agent is None:
        raise RuntimeError(
            f"Agent '{key}' was not initialized for the current mode. "
            f"Available agents: {list(agents.keys())}"
        )
    return agent


# ------------------------------------------------------------------
# Control-only nodes (no state writes)
# ------------------------------------------------------------------

def orchestrator_agent(state: MARVAState):
    logger.debug("Orchestrator received state (mode=%s)", state["mode"])
    return {
        "mode": state["mode"],
        "requirement": state.get("requirement"),
        "requirement_set": state.get("requirement_set"),
    }


def single_parallel_node(state: MARVAState):
    """Fan-out node for single-requirement validation."""
    logger.debug("Entering single parallel fan-out")
    return None


def group_parallel_node(state: MARVAState):
    """Fan-out node for group-level validation."""
    logger.debug("Entering group parallel fan-out")
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
    if max_workers is None:
        max_workers = len(agent_list)

    agent_names = [name for _, name in agent_list]
    logger.info("Starting parallel execution of %d agents: %s", len(agent_list), agent_names)
    overall_start = time.perf_counter()

    # Wrapper to time individual agent execution
    def timed_agent_run(agent, name):
        start = time.perf_counter()
        logger.debug("[%s] Started execution", name)
        result = agent.run(state)
        elapsed = time.perf_counter() - start
        logger.info("[%s] Completed in %.2fs", name, elapsed)
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
                logger.error("[%s] failed: %s", agent_name, e)
                raise

        overall_elapsed = time.perf_counter() - overall_start
        logger.info("Parallel execution completed in %.2fs (agents: %s)", overall_elapsed, agent_names)

        return merged_results


# ------------------------------------------------------------------
# Graph builder
# ------------------------------------------------------------------

def build_marva_s3_graph(agents: dict):
    logger.debug("Building S3 state graph")

    graph = StateGraph(MARVAState)

    # -------------------------------------------------
    # Nodes
    # -------------------------------------------------

    # Master orchestrator
    graph.add_node("orchestrator", orchestrator_agent)

    # Single-scope validation agents
    graph.add_node("atomicity", lambda s: _get_agent(agents, "atomicity").run(s))

    # Control / synchronization nodes
    graph.add_node("single_parallel", single_parallel_node)
    graph.add_node("group_parallel", group_parallel_node)

    # Parallel execution nodes
    def single_parallel_execution(state: MARVAState):
        agent_list = [
            (_get_agent(agents, "clarity"), "clarity"),
            (_get_agent(agents, "completion_single"), "completion_single")
        ]
        return execute_parallel_agents(state, agent_list, max_workers=2)

    def group_parallel_execution(state: MARVAState):
        agent_list = [
            (_get_agent(agents, "redundancy"), "redundancy"),
            (_get_agent(agents, "completion_group"), "completion_group"),
            (_get_agent(agents, "consistency_group"), "consistency_group")
        ]
        return execute_parallel_agents(state, agent_list, max_workers=3)

    graph.add_node("single_parallel_exec", single_parallel_execution)
    graph.add_node("group_parallel_exec", group_parallel_execution)

    # Decision agent
    graph.add_node("decision", lambda s: _get_agent(agents, "decision").run(s))

    # -------------------------------------------------
    # Entry point
    # -------------------------------------------------

    graph.set_entry_point("orchestrator")

    # -------------------------------------------------
    # Orchestrator routing (flow control)
    # -------------------------------------------------

    def orchestrator_router(state: MARVAState):
        mode = state["mode"]
        if mode == "single":
            logger.debug("Orchestrator routing to 'atomicity' (mode=single)")
            return "atomicity"
        if mode == "group":
            logger.debug("Orchestrator routing to 'group_parallel' (mode=group)")
            return "group_parallel"
        raise ValueError(f"Unknown mode: {mode}")

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
        status = state["atomicity"].status.upper()
        if status == "FAIL":
            logger.warning("Atomicity FAILED — skipping to decision (hard gate)")
            # return "decision"
            return "single_parallel"  # For now, we still run the parallel agents to gather more info for decision
        logger.debug("Atomicity passed (%s) — continuing to parallel agents", status)
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

    logger.debug("S3 state graph built successfully")
    return graph
