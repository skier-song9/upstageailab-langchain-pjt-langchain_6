def mock_db_select(paper_title: str) -> dict | None:
    """DB에서 논문을 조회하는 함수. 기본적으로 "논문의 제목"을 바탕으로 일치하는 논문을 찾는 로직으로 구현한다.
    (사용자가 읽은 논문이니 full title을 입력해줄 것으로 기대)

    1. DB에서 일치하는 논문 또는 유사한 제목의 논문을 찾아 반환한다.
    
    :param paper_title str:
    """
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
