from typing import TypedDict, List

class GraphState(TypedDict):
    """Represents the state of our graph."""
    initial_query: str
    sbp_found: bool # sbp (selected base paper)
    sbp_title: str # full title of sbp
    retrieved_docs: List[str]
    answer: str

    ### 김정빈 ###
    paper_search_result: dict | None
    ### 송규헌 ###


    ### 이나경 ###


    ### 조선미 ###

    ### 편아현 ###
