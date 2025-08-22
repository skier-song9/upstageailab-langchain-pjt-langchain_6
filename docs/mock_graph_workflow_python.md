## Project LangGraph Workflow
```python
import os
from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END

# --- 0. 환경 설정 (API 키 등) ---
# 실제 사용 시 API 키를 설정해야 합니다.
# os.environ["LANGCHAIN_API_KEY"] = "YOUR_API_KEY"
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# --- 1. 모의(Mock) 함수 정의 ---
# 실제 구현 시 이 부분들을 실제 DB 접속, API 호출, LLM 호출 로직으로 대체해야 합니다.

def mock_db_select(paper_title: str) -> dict | None:
    """DB에서 논문을 조회하는 모의 함수. 'Graph RAG' 논문만 찾을 수 있다고 가정합니다."""
    print(f"📄 DB 조회: '{paper_title}'")
    if "graph rag" in paper_title.lower():
        return {"title": "Graph RAG", "is_sbp": True, "details": "Graph RAG에 대한 상세 정보"}
    return None

def mock_web_search(paper_title: str) -> dict | None:
    """웹 검색 API를 호출하는 모의 함수."""
    print(f"🌐 웹 검색: '{paper_title}'")
    if "graph rag" in paper_title.lower():
        return {"title": "Graph RAG", "source": "Web", "details": "웹에서 찾은 Graph RAG 정보"}
    return None

def mock_db_insert(paper_info: dict):
    """DB에 논문 정보를 삽입하는 모의 함수."""
    print(f"💾 DB에 삽입: '{paper_info['title']}'")
    # 실제로는 DB에 저장하는 로직이 들어갑니다.
    pass

def mock_rag_retrieval(paper_title: str) -> List[str]:
    """Vector Store에서 관련 문서를 검색하는 RAG Retriever 모의 함수."""
    print(f"🔍 Vector Store 검색 (Retrieve): '{paper_title}' 기반 후속 논문")
    return ["후속 논문 A (from Vector Store)", "후속 논문 B (from Vector Store)"]

def mock_db_follow_up_select(paper_title: str) -> List[str]:
    """DB에서 인용 관계의 후속 논문을 조회하는 모의 함수."""
    print(f"🔍 DB 인용관계 검색 (Select): '{paper_title}' 인용 논문")
    return ["후속 논문 C (from DB)", "후속 논문 D (from DB)"]

def mock_llm_generate(context: List[str]) -> str:
    """검색된 문서를 바탕으로 최종 답변을 생성하는 LLM 모의 함수."""
    print("🤖 LLM 답변 생성 중...")
    context_str = "\n - ".join(context)
    return f"요청하신 논문의 주요 후속 연구는 다음과 같습니다:\n - {context_str}"


# --- 2. Graph 상태 정의 ---
# 그래프의 각 노드를 거치며 데이터가 저장되고 업데이트되는 상태 객체입니다.
class GraphState(TypedDict):
    """Represents the state of our graph."""
    initial_query: str
    sbp_found: bool
    sbp_title: str
    retrieved_docs: List[str]
    answer: str

# --- 3. LangGraph 노드 함수 정의 ---

def select_paper_node(state: GraphState) -> GraphState:
    """:param state: The current graph state. :return: New state with DB query result."""
    print("\n--- 노드 실행: select_paper_node ---")
    query = state["initial_query"]
    paper_info = mock_db_select(query)
    
    if paper_info and paper_info["is_sbp"]:
        return {"sbp_found": True, "sbp_title": paper_info["title"]}
    else:
        return {"sbp_found": False, "sbp_title": ""}

def web_search_node(state: GraphState) -> GraphState:
    """:param state: The current graph state. :return: New state with web search result."""
    print("\n--- 노드 실행: web_search_node ---")
    query = state["initial_query"]
    paper_info = mock_web_search(query)
    # 실제로는 이 정보를 다음 노드로 넘겨주기 위해 state에 저장해야 합니다.
    # 여기서는 insert_paper_node에서 다시 검색하는 간단한 형태로 구현합니다.
    return {}

def insert_paper_node(state: GraphState) -> GraphState:
    """:param state: The current graph state. :return: An empty dictionary as it only performs an action."""
    print("\n--- 노드 실행: insert_paper_node ---")
    query = state["initial_query"]
    paper_info = mock_web_search(query)
    if paper_info:
        mock_db_insert(paper_info)
    # 이 노드는 상태를 직접 변경하지 않고, DB에 삽입 후 루프를 통해 select_paper_node로 돌아갑니다.
    return {}
    
def retrieve_and_select_node(state: GraphState) -> GraphState:
    """:param state: The current graph state. :return: New state with retrieved documents."""
    print("\n--- 노드 실행: retrieve_and_select_node ---")
    sbp_title = state["sbp_title"]
    
    # 두 종류의 검색을 병렬로 수행 (실제로는 병렬 처리 라이브러리 사용 가능)
    retrieved_docs = mock_rag_retrieval(sbp_title)
    db_follow_up_docs = mock_db_follow_up_select(sbp_title)
    
    # 결과 병합 및 필터링 (여기서는 단순 합치기)
    all_docs = retrieved_docs + db_follow_up_docs
    return {"retrieved_docs": all_docs}

def generate_answer_node(state: GraphState) -> GraphState:
    """:param state: The current graph state. :return: New state with the final answer."""
    print("\n--- 노드 실행: generate_answer_node ---")
    context = state["retrieved_docs"]
    answer = mock_llm_generate(context)
    return {"answer": answer}

# --- 4. 조건부 엣지(Edge) 함수 정의 ---

def should_search_web(state: GraphState) -> str:
    """:param state: The current graph state. :return: The name of the next node to call."""
    print("\n--- 조건 분기: should_search_web ---")
    if state["sbp_found"]:
        print("✅ SBP 발견! Phase 2로 진행합니다.")
        return "retrieve_and_select"
    else:
        print("❌ SBP 미발견. 웹 검색을 시작합니다.")
        return "web_search"

# --- 5. 그래프 구성 및 컴파일 ---

workflow = StateGraph(GraphState)

# 노드 추가
workflow.add_node("select_paper", select_paper_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("insert_paper", insert_paper_node)
workflow.add_node("retrieve_and_select", retrieve_and_select_node)
workflow.add_node("generate_answer", generate_answer_node)

# 엣지 연결
workflow.set_entry_point("select_paper")
workflow.add_edge("web_search", "insert_paper")
workflow.add_edge("insert_paper", "select_paper") # 루프 형성
workflow.add_edge("retrieve_and_select", "generate_answer")
workflow.add_edge("generate_answer", END)

# 조건부 엣지 연결
workflow.add_conditional_edges(
    "select_paper",
    should_search_web,
    {
        "web_search": "web_search",
        "retrieve_and_select": "retrieve_and_select",
    },
)

# 그래프 컴파일
app = workflow.compile()

# --- 6. 그래프 실행 ---

# 시나리오 1: DB에 SBP가 바로 없는 경우 (웹 검색 -> 삽입 -> 재검색 -> Phase 2)
print("===== 시나리오 1: 'Some other paper' 검색 시작 =====")
inputs = {"initial_query": "Some other paper"}
for event in app.stream(inputs):
    for key, value in event.items():
        print(f"--- 실행된 노드: {key} ---")
        print(value)
print("\n최종 결과:", app.invoke(inputs)["answer"])


# 시나리오 2: DB에 SBP가 바로 있는 경우 (Phase 2로 바로 진행)
print("\n\n===== 시나리오 2: 'Graph RAG' 검색 시작 =====")
inputs = {"initial_query": "Graph RAG"}
final_state = app.invoke(inputs)
print("\n최종 결과:", final_state["answer"])
```