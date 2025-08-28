from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages
from langchain_core.documents import Document

class GraphState(TypedDict):
    """Represents the state of our graph."""
    initial_query: str # 사용자가 입력한 논문 제목
    sbp_found: bool # sbp (selected base paper)
    sbp_title: str # full title of sbp
    retrieved_docs: List[Document]
    answer: str

    ### 김정빈 ###
    paper_search_result: dict | None
    is_chat_mode: bool
    rag_judgement: str
    messages: Annotated[list, add_messages]
    ### 송규헌 ###
    thread_id: str # 각 대화 세션을 식별하는 ID
    question: str # Phase 2 에서 사용자가 입력한 프롬프트
    history: str

    ### 이나경 ###


    ### 조선미 ###

    ### 편아현 ###
    