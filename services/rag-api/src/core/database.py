def mock_db_select(paper_title: str) -> dict | None:
    """DB에서 논문을 조회하는 모의 함수. 'Graph RAG' 논문만 찾을 수 있다고 가정합니다."""
    print(f"📄 DB 조회: '{paper_title}'")
    if "graph rag" in paper_title.lower():
        return {"title": "Graph RAG", "is_sbp": True, "details": "Graph RAG에 대한 상세 정보"}
    return None

def mock_db_insert(paper_info: dict):
    """DB에 논문 정보를 삽입하는 모의 함수."""
    print(f"💾 DB에 삽입: '{paper_info['title']}'")
    # 실제로는 DB에 저장하는 로직이 들어갑니다.
    pass

def mock_db_follow_up_select(paper_title: str) -> list[str]:
    """DB에서 인용 관계의 후속 논문을 조회하는 모의 함수."""
    print(f"🔍 DB 인용관계 검색 (Select): '{paper_title}' 인용 논문")
    return ["후속 논문 C (from DB)", "후속 논문 D (from DB)"]
