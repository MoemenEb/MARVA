from typing import TypedDict, Dict, Any
from typing_extensions import Annotated
import operator


class MARVAState(TypedDict, total=False):
    """
    Explicit LangGraph state for MARVA S3.

    Each agent owns exactly one key.
    All keys are mergeable.
    """

    # routing context (read-only)
    mode: str

    # single-scope agents
    atomicity: Dict[str, Any]
    clarity: Dict[str, Any]
    completion_single: Dict[str, Any]
    consistency_single: Dict[str, Any]

    # group-scope agents
    redundancy: Dict[str, Any]
    completion_group: Dict[str, Any]
    consistency_group: Dict[str, Any]

    # final decision
    decision: Dict[str, Any]
