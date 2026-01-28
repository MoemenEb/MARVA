from typing import TypedDict, Dict, Any, List
from typing_extensions import Annotated
from entity.requirement import Requirement


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
    group: Annotated[List[Requirement], replace]

    # ----------------------------
    # Single-scope agent outputs
    # ----------------------------
    atomicity: Annotated[Dict[str, Any], replace]
    clarity: Annotated[Dict[str, Any], replace]
    completion_single: Annotated[Dict[str, Any], replace]
    # consistency_single: Annotated[Dict[str, Any], replace]

    # ----------------------------
    # Group-scope agent outputs
    # ----------------------------
    redundancy: Annotated[Dict[str, Any], replace]
    completion_group: Annotated[Dict[str, Any], replace]
    consistency_group: Annotated[Dict[str, Any], replace]

    # ----------------------------
    # Final decision
    # ----------------------------
    decision: Annotated[Dict[str, Any], replace]
