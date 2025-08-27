from typing import List

def mock_rag_retrieval(paper_title: str) -> List[str]:
    """Vector Store에서 관련 문서를 검색하는 RAG Retriever 모의 함수."""
    print(f"🔍 Vector Store 검색 (Retrieve): '{paper_title}' 기반 후속 논문")
    return ["후속 논문 A (from Vector Store)", "후속 논문 B (from Vector Store)"]
