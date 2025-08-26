from typing import TypedDict, List

class GraphState(TypedDict):
    """Represents the state of our graph."""
    initial_query: str
    base_paper: str # 사용자가 읽은 논문
    sbp_found: bool # sbp (selected base paper)
    sbp_title: str # full title of sbp
    retrieved_docs: List[str]
    answer: str
    ### 김정빈 ###


    ### 송규헌 ###


    ### 이나경 ###


    ### 조선미 ###

    ### 편아현 ###
