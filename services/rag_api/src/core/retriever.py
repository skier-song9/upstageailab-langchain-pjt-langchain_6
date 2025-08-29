import functools
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_upstage import ChatUpstage
from langchain_tavily import TavilySearch
from langchain_community.tools.tavily_search import TavilyAnswer # TavilyAnswer는 질문에 대한 직접적인 답변을 생성

import os 
from dotenv import load_dotenv
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')) # 5단계 위로 이동
load_dotenv(os.path.join(ROOT_DIR, '.env'))
UPSTAGE_API_KEY = os.getenv('UPSTAGE_API_KEY')
TAVILY_SEARCH = os.getenv('TAVILY_SEARCH')

def mock_rag_retrieval(paper_title: str) -> List[str]:
    """Vector Store에서 관련 문서를 검색하는 RAG Retriever 모의 함수."""
    print(f"🔍 Vector Store 검색 (Retrieve): '{paper_title}' 기반 후속 논문")
    return ["후속 논문 A (from Vector Store)", "후속 논문 B (from Vector Store)"]

# 1. 키워드 추출을 위한 데이터 구조 정의 (Pydantic 모델)
class Keywords(BaseModel):
    """A list of keywords extracted from the user's question."""
    keywords: List[str] = Field(description="사용자 질문에서 추출된 핵심 키워드 리스트")


def augment_prompt(question: str, llm_api_key: str, tavily_search_key: str) -> str:
    """사용자 prompt에서 키워드를 추출하여 tavily search로 증강한 후, 영어로 번역하여 반환하는 함수
    1. Upstage의 solar-mini LLM 모델을 사용해 question으로부터 키워드를 추출하여라. 이때, LLM의 답변이 List[str] 이 되도록 형식을 제한하는 프롬프트를 잘 작성하여라. 또는 Langchain에서 OutputFixingParser와 같은 클래스를 활용하여 출력 형식을 제한하여라.

    2. 1번에서 구한 키워드 리스트의 각 단어들을 tavily-search 를 사용해 각 키워드에 대해 한 줄짜리 부가설명을 구하여라.

    3. question의 키워드에 2번에서 구한 부가설명을 키워드 단어 뒤에 괄호 안에 추가하여라.
    e.g. 기존 question : Downstream task에 대해 모델의 재사용성을 향상시킨 후속논문을 알려줘.
        augmented_question : Downstream task(AI와 머신러닝에서 사전 학습된 모델을 특정 작업에 맞게 조정하여 활용하는 과정)에 대해 모델의 재사용성(이미 개발된 머신러닝 모델을 다른 문제나 환경에 적용하여 활용하는 것을 의미)을 향상시킨 후속논문을 알려줘.

    4. Upstage의 solar-pro2 LLM 모델을 사용해 question을 영어로 번역하여 return

    :param str question: user prompt
    :param str llm_api_key: Upstage API Key
    :return str: augmented prompt & translated to English
    """
    # 1. solar-mini 로 키워드 추출
    llm_mini = ChatUpstage(api_key=llm_api_key, model='solar-pro2')
    # JSON 형식으로 출력을 파싱하는 파서 설정
    parser = JsonOutputParser(pydantic_object=Keywords)
    # 키워드 추출을 위한 프롬프트 템플릿 정의
    keyword_prompt = ChatPromptTemplate.from_template(
        """You are an expert in extracting keywords from a text.
Extract the main keywords from the following user question.
Your output must be a JSON object with a single key 'keywords' containing a list of the extracted keywords.
Exclude keywords related to 'follow-up papers', '후속 논문'.

Question: {question}

{format_instructions}"""
    )
    # LCEL을 사용해 키워드 추출 체인 구성
    keyword_chain = keyword_prompt | llm_mini | parser
    # 체인 실행
    response = keyword_chain.invoke({
        "question": question,
        "format_instructions": parser.get_format_instructions()
    })
    keywords = response['keywords']
    print(f"✅ 추출된 키워드: {keywords}")

    # --- 2단계: Tavily Search로 각 키워드에 대한 부가설명 검색 ---
    # Tavily Search 도구 초기화
    search = TavilySearch(tavily_api_key=tavily_search_key, max_results=1)
    keyword_definitions = {}

    # print("\n--- 2. 키워드 정의 검색 중... ---")
    for keyword in keywords:
        # 각 키워드에 대한 한 줄 정의를 얻기 위해 구체적인 쿼리 생성
        search_query = f'Machine Learning, Deep Learning, AI와 관련하여 "{keyword}"에 대한 한 줄 정의'
        search_result = search.invoke(search_query)
        
        # 검색 결과가 있고, content 키가 존재하면 정의 추출
        if search_result and 'content' in search_result['results'][0]:
            definition = search_result['results'][0]['content']
            keyword_definitions[keyword] = definition
        else:
            keyword_definitions[keyword] = ""

    # --- 3단계: keyword:definition pair formatting ---
    keydef_pair = ""
    for keyword, definition in keyword_definitions.items():
        keydef_pair += f"- {keyword}: {definition}\n"
    print("⚙️key def pair:", keydef_pair)

    # --- 4단계: Upstage solar-pro2 LLM을 사용하여 영어로 번역 ---
    
    # solar-pro2 모델 초기화
    llm_pro = ChatUpstage(api_key=llm_api_key, model="solar-pro2")
    
    # 번역을 위한 프롬프트 템플릿 정의
    translate_prompt = ChatPromptTemplate.from_messages([
        ("system","You are a professional translator. Translate the following Korean text into English."),
        ('user',"""Translate the given text into English following these rules:
1. You will be given a list of keyword: definition pairs. Whenever a keyword appears in the text and you translate it into English, you must include its corresponding definition in parentheses immediately after the translated keyword.
Example: If you are given key1: def1 and the text says “key1을 따르는”, you should translate it as key1(def1) is.
2. Your response must contain only the translated English text, with no additional explanations or extra words.

Keyword:Definition pairs:
{keydef_pair}
Text: {text_to_translate}""")]
    )
    
    # 출력을 문자열로 파싱하는 파서 설정
    output_parser = StrOutputParser()
    
    # LCEL을 사용해 번역 체인 구성
    translate_chain = translate_prompt | llm_pro | output_parser
    
    print("\n--- 4. 영어로 번역 중... ---")
    # 체인 실행
    final_result = translate_chain.invoke({"keydef_pair":keydef_pair, "text_to_translate": question})
    print("✅ 번역 완료!")
    return final_result


if __name__ == "__main__":
    import os 
    from dotenv import load_dotenv
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')) # 5단계 위로 이동
    print(ROOT_DIR)
    load_dotenv(os.path.join(ROOT_DIR, '.env'))
    UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
    TAVILY_SEARCH = os.getenv("TAVILY_SEARCH")
    print(UPSTAGE_API_KEY, TAVILY_SEARCH)

    test_question = "Downstream task에 대해 모델의 재사용성을 향상시킨 후속논문을 알려줘."
    print(f"\n--- 원본 질문 --- \n{test_question}\n")

    # 함수 실행
    augmented_and_translated_prompt = augment_prompt(
        question=test_question,
        llm_api_key=UPSTAGE_API_KEY,
        tavily_search_key=TAVILY_SEARCH
    )

    print("\n========================================")
    print("    ✨ 최종 번역된 증강 프롬프트 ✨")
    print("========================================")
    print(augmented_and_translated_prompt)