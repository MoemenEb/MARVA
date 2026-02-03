from typing import TypedDict
from typing_extensions import Annotated
from entity.requirement import Requirement
from entity.requirement_set import RequirementSet
from entity.agent import AgentResult


def replace(_, new):
    return new


class MARVAState(TypedDict, total=False):
    """
    Explicit LangGraph state for MARVA S3.
    """

    # ----------------------------
    # Routing context
    # ----------------------------
    mode: str

    # ----------------------------
    # INPUT CHANNELS (IMMUTABLE)
    # ----------------------------
    requirement: Annotated[Requirement, replace]
    requirement_set: Annotated[RequirementSet, replace]

    # ----------------------------
    # Single-scope agent outputs
    # ----------------------------
    atomicity: Annotated[AgentResult, replace]
    clarity: Annotated[AgentResult, replace]
    completion_single: Annotated[AgentResult, replace]

    # ----------------------------
    # Group-scope agent outputs
    # ----------------------------
    redundancy: Annotated[AgentResult, replace]
    completion_group: Annotated[AgentResult, replace]
    consistency_group: Annotated[AgentResult, replace]

    # ----------------------------
    # Final decision
    # ----------------------------
    decision: Annotated[AgentResult, replace]
