from typing import List
import re
from collections import defaultdict

def mock_llm_generate(context: List[str]) -> str:
    """
    검색된 문서를 바탕으로 최종 답변을 생성하는 LLM 모의 함수.
    논문들을 분석하여 구조화된 답변을 생성합니다.
    
    :param context: 검색된 후속 연구 논문들의 리스트
    :return: 구조화된 답변 문자열
    """
    print("🤖 LLM 답변 생성 중...")
    
    if not context:
        return "죄송합니다. 관련된 후속 연구를 찾을 수 없습니다."
    
    # 논문들을 분석하여 구조화
    analyzed_papers = analyze_papers(context)
    
    # 구조화된 답변 생성
    answer = generate_structured_answer(analyzed_papers)
    
    return answer

def analyze_papers(papers: List[str]) -> dict:
    """
    논문 리스트를 분석하여 카테고리별로 분류하고 중요도 분석
    
    :param papers: 논문 리스트
    :return: 분석된 논문 정보
    """
    analysis = {
        'total_count': len(papers),
        'categories': defaultdict(list),
        'key_papers': [],
        'research_trends': []
    }
    
    for paper in papers:
        # 논문 제목에서 키워드 추출
        keywords = extract_keywords(paper)
        
        # 카테고리 분류
        category = classify_paper_category(paper, keywords)
        analysis['categories'][category].append(paper)
        
        # 중요도 평가 (제목 길이, 특정 키워드 포함 여부 등)
        importance = evaluate_importance(paper, keywords)
        if importance > 0.7:  # 중요도가 높은 논문
            analysis['key_papers'].append((paper, importance))
    
    # 연구 트렌드 분석
    analysis['research_trends'] = analyze_research_trends(analysis['categories'])
    
    return analysis

def extract_keywords(paper: str) -> List[str]:
    """논문 제목에서 키워드 추출"""
    # 일반적인 AI/ML 키워드들
    ai_keywords = [
        'neural', 'network', 'learning', 'deep', 'transformer', 'attention',
        'bert', 'gpt', 'llm', 'rag', 'retrieval', 'generation', 'embedding',
        'vector', 'graph', 'nlp', 'computer vision', 'reinforcement',
        'optimization', 'architecture', 'model', 'algorithm'
    ]
    
    paper_lower = paper.lower()
    found_keywords = []
    
    for keyword in ai_keywords:
        if keyword in paper_lower:
            found_keywords.append(keyword)
    
    return found_keywords

def classify_paper_category(paper: str, keywords: List[str]) -> str:
    """논문을 카테고리별로 분류"""
    paper_lower = paper.lower()
    
    if any(word in paper_lower for word in ['transformer', 'attention', 'bert', 'gpt', 'llm']):
        return 'Language Models'
    elif any(word in paper_lower for word in ['rag', 'retrieval', 'generation']):
        return 'Retrieval-Augmented Generation'
    elif any(word in paper_lower for word in ['neural', 'network', 'deep']):
        return 'Neural Networks'
    elif any(word in paper_lower for word in ['graph', 'gnn']):
        return 'Graph Neural Networks'
    elif any(word in paper_lower for word in ['computer vision', 'image', 'vision']):
        return 'Computer Vision'
    elif any(word in paper_lower for word in ['reinforcement', 'rl']):
        return 'Reinforcement Learning'
    else:
        return 'General AI/ML'

def evaluate_importance(paper: str, keywords: List[str]) -> float:
    """논문의 중요도 평가 (0.0 ~ 1.0)"""
    importance = 0.0
    
    # 키워드 수에 따른 점수
    importance += min(len(keywords) * 0.1, 0.3)
    
    # 제목 길이에 따른 점수 (적당한 길이가 좋음)
    title_length = len(paper)
    if 20 <= title_length <= 80:
        importance += 0.2
    elif title_length > 80:
        importance += 0.1
    
    # 특정 중요 키워드에 대한 보너스
    if any(word in paper.lower() for word in ['survey', 'review', 'comprehensive']):
        importance += 0.3
    
    if any(word in paper.lower() for word in ['state-of-the-art', 'sota', 'breakthrough']):
        importance += 0.2
    
    return min(importance, 1.0)

def analyze_research_trends(categories: dict) -> List[str]:
    """연구 트렌드 분석"""
    trends = []
    
    # 가장 많은 논문이 있는 카테고리
    if categories:
        most_popular = max(categories.items(), key=lambda x: len(x[1]))
        trends.append(f"'{most_popular[0]}' 분야가 가장 활발하게 연구되고 있습니다.")
    
    # 다양한 분야가 연구되고 있는지
    if len(categories) >= 3:
        trends.append("다양한 AI 분야에서 후속 연구가 진행되고 있습니다.")
    elif len(categories) == 2:
        trends.append("주로 두 분야에서 후속 연구가 집중되고 있습니다.")
    else:
        trends.append("특정 분야에 집중된 후속 연구가 진행되고 있습니다.")
    
    return trends

def generate_structured_answer(analysis: dict) -> str:
    """구조화된 답변 생성"""
    answer_parts = []
    
    # 헤더
    answer_parts.append("🔍 **후속 연구 분석 결과**")
    answer_parts.append(f"총 {analysis['total_count']}개의 관련 논문을 발견했습니다.\n")
    
    # 연구 트렌드
    if analysis['research_trends']:
        answer_parts.append("📈 **연구 트렌드**")
        for trend in analysis['research_trends']:
            answer_parts.append(f"• {trend}")
        answer_parts.append("")
    
    # 카테고리별 분류
    if analysis['categories']:
        answer_parts.append("📚 **분야별 분류**")
        for category, papers in analysis['categories'].items():
            answer_parts.append(f"• **{category}**: {len(papers)}개 논문")
        answer_parts.append("")
    
    # 주요 논문들
    if analysis['key_papers']:
        answer_parts.append("⭐ **주요 논문**")
        # 중요도 순으로 정렬
        sorted_papers = sorted(analysis['key_papers'], key=lambda x: x[1], reverse=True)
        for paper, importance in sorted_papers[:5]:  # 상위 5개만
            importance_stars = "⭐" * int(importance * 5)
            answer_parts.append(f"• {importance_stars} {paper}")
        answer_parts.append("")
    
    # 상세 논문 목록
    answer_parts.append("📖 **전체 논문 목록**")
    for i, paper in enumerate(analysis['categories'].values(), 1):
        for p in paper:
            answer_parts.append(f"{i}. {p}")
            i += 1
    
    return "\n".join(answer_parts)
