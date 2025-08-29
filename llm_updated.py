from importlib import metadata
from typing import List, Dict, Any, Optional
import re
from collections import defaultdict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

def format_context(context: List[Document]) -> str:
    """
    LangChain Document 리스트를 LLM 프롬프트에 적합한 단일 문자열로 변환합니다.
    각 논문의 출처(로컬 DB, Tavily, OpenAlex)를 함께 표시합니다.
    
    Args:
        context: 'title'을 metadata에 포함하는 Document 객체 리스트
        
    Returns:
        각 논문 정보와 출처가 포함된 전체 문자열
    """
    context_parts = []
    for i, doc in enumerate(context):
        # doc.metadata에서 'title'을, doc.page_content에서 초록을 가져옵니다.
        title = doc.metadata.get('title', 'No Title Provided')
        abstract = doc.page_content
        
        # 출처 판별하기
        source = determine_paper_source(abstract)
        
        context_parts.append(f"--- Paper {i+1} ({source}) ---\nTitle: {title}\nAbstract: {abstract}")
    
    return "\n\n".join(context_parts)

def determine_paper_source(content: str) -> str:
    """
    논문 내용을 분석하여 어느 소스에서 가져왔는지 판별합니다.
    
    :param content: 논문의 내용 (page_content)
    :return: 출처 문자열
    """
    # 로컬 데이터베이스 식별 패턴
    if "📄" in content and "유사도:" in content and "인용수:" in content:
        return "로컬 벡터 DB"
    
    # OpenAlex 식별 패턴
    elif "📚" in content and "저자:" in content and ("📊 인용수:" in content or "📝 요약:" in content):
        return "OpenAlex 학술 검색"
    
    # Tavily 식별 패턴
    elif "🤖 AI 요약:" in content or "🔗" in content or "🌐" in content:
        return "Tavily 웹 검색"
    
    # 구체적인 패턴으로 재시도
    elif "**로컬 데이터베이스 검색 결과:**" in content:
        return "로컬 벡터 DB"
    elif "**OpenAlex 학술 검색 결과:**" in content:
        return "OpenAlex 학술 검색" 
    elif "**웹 검색 결과 (최신 정보):**" in content:
        return "Tavily 웹 검색"
    
    # 기본값
    else:
        return "미분류"

def llm_generate(question: str, context: List[Document], llm_api_key: str) -> str:
    """
    검색된 문서를 바탕으로 최종 답변을 생성하는 LLM 함수.
    논문들을 분석하여 구조화된 답변을 생성합니다.
    1. Document에는 title(논문 제목)과 content(논문 abstract 내용)가 있다.
    2. context의 길이가 0이면 "검색된 후속논문이 없다"는 내용의 답변을 반환하여라. (LLM 사용 금지)
    3. format_context 함수를 통해 context의 각 Document들을 `title: 논문 제목, abstract: 논문 abstract 내용` 으로 재구성하여 context_str 에 저장하여라.
    4. prompt를 사용하여 LangChain 문법에 따라 LLM(openai GPT-3.5 Turbo)으로부터 답변을 생성하여라.
    
    :param str question: 사용자가 입력한 프롬프트
    :param List[Document] context: 검색된 후속 연구 논문들의 리스트
    :param llm_api_key: OpenAI API 키
    :return str: 구조화된 답변 문자열
    """
    print("🤖 LLM 답변 생성 중...")
    print(f"🔍 DEBUG: 받은 질문 = '{question}'")
    print(f"📄 DEBUG: 컨텍스트 개수 = {len(context)}")
    
    # 2. context 리스트가 비어있는지 확인합니다.
    if not context:
        print("ℹ️ 컨텍스트가 비어있어 LLM을 호출하지 않고 기본 메시지를 반환합니다.")
        return "검색된 후속 논문이 없습니다. 다른 키워드로 검색해 보세요."
    
    # 3. context를 프롬프트에 넣기 좋은 단일 문자열로 formatting한다.
    context_str = format_context(context)

    # 4. LangChain 프롬프트 템플릿을 정의합니다.
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a world-class AI research assistant. Your primary goal is to provide a clear, structured, and insightful analysis of academic papers based on their titles and abstracts.
You must synthesize information from multiple sources and present it in an easy-to-digest format for researchers.
Your response must be well-organized, using Markdown for headings and lists.
"""
            ),
            (
                "human",
                """다음 연구 논문들을 기반으로 사용자의 질문에 답해주세요.

**사용자 질문:**
<question>
{question}
</question>

**제공된 맥락 (후속 연구 논문들):**
<context>
{context_str}
</context>

**답변 지침:**
1. **질문 유형을 내부적으로 판단하되, 판단 과정을 답변에 표시하지 마세요.**

2. **연구 관련 질문인 경우:**
   - **다양한 출처에서 균형있게 정보를 선택**하여 답변하세요.
   - **로컬 DB, OpenAlex, Tavily 각각에서 최소 1개씩**은 포함하여 답변하세요.
   - 각 논문에 대해 질문과 관련된 1-2줄 요약을 제공하고, **반드시 출처를 표시하세요**.
   - 출처 표시 방법:
     - 로컬 벡터 DB → 📚 (로컬 DB)
     - OpenAlex 학술 검색 → 🎓 (OpenAlex)
     - Tavily 웹 검색 → 🌐 (Tavily)
   - 질문에 따라 답변 스타일을 조정하세요:
     - "주요 후속 연구" → 연구 목록과 분야별 분류
     - "기술적 영향" → 구체적인 기술적 기여도 분석
     - "단점 보완" → 문제점과 해결책 중심
     - "핵심 혁신" → 혁신적 요소들 강조
     - "영향" → 파급 효과와 변화 중심
   - **출처별로 다양성을 확보하여** 포괄적인 답변을 제공하세요.

3. **일반적인 질문인 경우:**
   - 논문 내용을 언급하지 마세요.
   - 논문 추천이나 요약을 하지 마세요.
   - 간단하고 적절한 답변만 제공하세요.

**중요사항:**
- 답변에 "답변:", "질문 판단:" 등의 레이블을 붙이지 마세요.
- 내부 판단 과정을 사용자에게 보여주지 마세요.
- 바로 최종 답변만 제공하세요.
- 모든 답변은 한국어로 작성하세요.
- 답변은 명확하고 간결하게 작성하세요.
- 같은 내용을 반복하지 마세요.
- 답변이 완료되면 자연스럽게 종료하세요.
"""
            ),
        ]
    )

    # LLM 모델을 초기화합니다. (GPT-4o-mini 사용)
    llm = ChatOpenAI(
        model_name="gpt-4o-mini", 
        api_key=llm_api_key, 
        temperature=0.2,  # 일관성 있는 답변
        max_tokens=1000,  # 최대 토큰 제한
        frequency_penalty=0.5,  # 반복 패널티
        presence_penalty=0.3    # 존재 패널티
    )
    
    # LangChain Expression Language (LCEL)을 사용하여 체인을 구성합니다.
    # 1. 프롬프트 포맷팅 -> 2. LLM 호출 -> 3. 출력 파싱(문자열로)
    chain = prompt_template | llm | StrOutputParser()
    
    # 체인을 실행하여 답변을 생성합니다.
    answer = chain.invoke({
        "question": question,
        "context_str": context_str
    })
    
    return answer