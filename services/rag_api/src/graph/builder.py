import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.append(ROOT_DIR)

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from services.rag_api.src.graph.state import GraphState
from services.rag_api.src.graph.nodes import (
    select_paper_node,
    web_search_node,
    insert_paper_node,
    retrieve_and_select_node,
    generate_answer_node,
    should_search_web,
    rag_judge_node,
    rag_condition,
)

from services.rag_api.src.core.get_emb import get_emb_model

def build_graph():
    checkpointer = MemorySaver()
    workflow = StateGraph(GraphState)

    workflow.add_node("select_paper", select_paper_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("insert_paper", insert_paper_node)
    workflow.add_node("retrieve_and_select", retrieve_and_select_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("rag_judge", rag_judge_node)
    
    workflow.set_entry_point("select_paper")
    workflow.add_edge("web_search", "insert_paper")
    workflow.add_edge("insert_paper", "select_paper")
    workflow.add_edge("retrieve_and_select", "generate_answer")
    workflow.add_edge("generate_answer", END)

    workflow.add_conditional_edges(
        "select_paper",
        should_search_web,
        {
            "web_search": "web_search",
            "rag_judge": "rag_judge",
        },
    )

    workflow.add_conditional_edges(
        "rag_judge",
        rag_condition,
        {
          "retrieve_and_select": "retrieve_and_select",
          "generate_answer": "generate_answer",
        },
    )


    model = get_emb_model() # 임베딩 모델 로드(캐시 적용되어 이후 노드들에서는 로드 X)

    # Checkpointer와 함께 그래프를 컴파일하고, select_paper 이후에 중단점을 설정합니다.
    return workflow.compile(
        checkpointer=checkpointer
        # interrupt_after=["select_paper"] # select_paper 노드 실행 후 사용자 입력을 위해 대기
    )

if __name__ == "__main__":
    graph = build_graph()
    graph.get_graph().draw_png("graph.png")