from langgraph.graph import StateGraph, END
from .state import GraphState
from .nodes import (
    select_paper_node,
    web_search_node,
    insert_paper_node,
    retrieve_and_select_node,
    generate_answer_node,
    should_search_web,
)

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("select_paper", select_paper_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("insert_paper", insert_paper_node)
    workflow.add_node("retrieve_and_select", retrieve_and_select_node)
    workflow.add_node("generate_answer", generate_answer_node)

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
            "retrieve_and_select": "retrieve_and_select",
        },
    )

    return workflow.compile()
