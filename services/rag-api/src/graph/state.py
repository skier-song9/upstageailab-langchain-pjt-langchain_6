from typing import TypedDict, List

class GraphState(TypedDict):
    """Represents the state of our graph."""
    initial_query: str
    sbp_found: bool
    sbp_title: str
    retrieved_docs: List[str]
    answer: str
