from .state import GraphState
from ..core.database import mock_db_select, mock_db_insert, mock_db_follow_up_select
from ..core.source_api import mock_web_search
from ..core.retriever import mock_rag_retrieval
from ..core.llm import mock_llm_generate

def select_paper_node(state: GraphState):
    """:param state: The current graph state. :return: New state with DB query result."""
    print("\n--- 노드 실행: select_paper_node ---")
    query = state["initial_query"]
    paper_info = mock_db_select(query)
    
    if paper_info and paper_info["is_sbp"]:
        return {"sbp_found": True, "sbp_title": paper_info["title"]}
    else:
        return {"sbp_found": False, "sbp_title": ""}

def web_search_node(state: GraphState):
    """:param state: The current graph state. :return: New state with web search result."""
    print("\n--- 노드 실행: web_search_node ---")
    query = state["initial_query"]
    paper_info = mock_web_search(query)
    return {}

def insert_paper_node(state: GraphState):
    """:param state: The current graph state. :return: An empty dictionary as it only performs an action."""
    print("\n--- 노드 실행: insert_paper_node ---")
    query = state["initial_query"]
    paper_info = mock_web_search(query)
    if paper_info:
        mock_db_insert(paper_info)
    return {}
    
def retrieve_and_select_node(state: GraphState):
    """:param state: The current graph state. :return: New state with retrieved documents."""
    print("\n--- 노드 실행: retrieve_and_select_node ---")
    sbp_title = state["sbp_title"]
    
    retrieved_docs = mock_rag_retrieval(sbp_title)
    db_follow_up_docs = mock_db_follow_up_select(sbp_title)
    
    all_docs = retrieved_docs + db_follow_up_docs
    return {"retrieved_docs": all_docs}

def generate_answer_node(state: GraphState):
    """:param state: The current graph state. :return: New state with the final answer."""
    print("\n--- 노드 실행: generate_answer_node ---")
    context = state["retrieved_docs"]
    answer = mock_llm_generate(context)
    return {"answer": answer}

def should_search_web(state: GraphState) -> str:
    """:param state: The current graph state. :return: The name of the next node to call."""
    print("\n--- 조건 분기: should_search_web ---")
    if state["sbp_found"]:
        print("✅ SBP 발견! Phase 2로 진행합니다.")
        return "retrieve_and_select"
    else:
        print("❌ SBP 미발견. 웹 검색을 시작합니다.")
        return "web_search"