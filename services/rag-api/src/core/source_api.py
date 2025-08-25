def mock_web_search(paper_title: str) -> dict | None:
    """웹 검색 API를 호출하는 모의 함수."""
    print(f"🌐 웹 검색: '{paper_title}'")
    if "graph rag" in paper_title.lower():
        return {"title": "Graph RAG", "source": "Web", "details": "웹에서 찾은 Graph RAG 정보"}
    return None
