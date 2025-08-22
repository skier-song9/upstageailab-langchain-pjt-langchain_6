from typing import List

def mock_llm_generate(context: List[str]) -> str:
    """검색된 문서를 바탕으로 최종 답변을 생성하는 LLM 모의 함수."""
    print("🤖 LLM 답변 생성 중...")
    context_str = "\n - ".join(context)
    return f"요청하신 논문의 주요 후속 연구는 다음과 같습니다:\n - {context_str}"
