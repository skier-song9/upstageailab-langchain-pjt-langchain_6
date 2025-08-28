from importlib import metadata
from typing import List
import re
from collections import defaultdict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

def format_context(context: List[Document]) -> str:
    """
    LangChain Document 리스트를 LLM 프롬프트에 적합한 단일 문자열로 변환합니다.
    
    :param context: 'title'을 metadata에 포함하는 Document 객체 리스트
    :return: 각 논문 정보가 포함된 전체 문자열
    """
    context_parts = []
    for i, doc in enumerate(context):
        # doc.metadata에서 'title'을, doc.page_content에서 초록을 가져옵니다.
        title = doc.metadata.get('title', 'No Title Provided')
        abstract = doc.page_content
        context_parts.append(f"--- Paper {i+1} ---\nTitle: {title}\nAbstract: {abstract}")
    
    return "\n\n".join(context_parts)

def mock_llm_generate(question: str, context: List[Document], llm_api_key: str) -> str:
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
                """Based on the following research papers, please answer my question.

**User's Question:**
<question>
{question}
</question>

**Provided Context (Follow-up Papers):**
<context>
{context_str}
</context>

**Your Task:**
Directly answer the user's question. Synthesize the key findings from all papers in the context to support your answer. Provide a brief summary in one or two lines for each papers. Focus on what is necessary to answer the question and finally please recommend the best one to read first out of context papers.

**IMPORTANT**: Your final output and all content must be written in **Korean**.
"""
            ),
        ]
    )

    # LLM 모델을 초기화합니다. (GPT-3.5 Turbo 사용)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=llm_api_key, temperature=0.2)
    
    # LangChain Expression Language (LCEL)을 사용하여 체인을 구성합니다.
    # 1. 프롬프트 포맷팅 -> 2. LLM 호출 -> 3. 출력 파싱(문자열로)
    chain = prompt_template | llm | StrOutputParser()
    
    # 체인을 실행하여 답변을 생성합니다.
    answer = chain.invoke({
        "question": question,
        "context_str": context_str
    })
    
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


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')) # 5단계 위로 이동
    print(ROOT_DIR)
    load_dotenv(os.path.join(ROOT_DIR, '.env'))
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    print(OPENAI_API_KEY)
    p1 = Document(metadata={'title':'Attention is all you need'}, page_content="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.")
    p2 = Document(metadata={'title':'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding'}, page_content='We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).')
    p3 = Document(metadata={'title':'BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension'}, page_content='We present BART, a denoising autoencoder for pretraining sequence-to-sequence models. BART is trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original text. It uses a standard Tranformer-based neural machine translation architecture which, despite its simplicity, can be seen as generalizing BERT (due to the bidirectional encoder), GPT (with the left-to-right decoder), and many other more recent pretraining schemes. We evaluate a number of noising approaches, finding the best performance by both randomly shuffling the order of the original sentences and using a novel in-filling scheme, where spans of text are replaced with a single mask token. BART is particularly effective when fine tuned for text generation but also works well for comprehension tasks. It matches the performance of RoBERTa with comparable training resources on GLUE and SQuAD, achieves new state-of-the-art results on a range of abstractive dialogue, question answering, and summarization tasks, with gains of up to 6 ROUGE. BART also provides a 1.1 BLEU increase over a back-translation system for machine translation, with only target language pretraining. We also report ablation experiments that replicate other pretraining schemes within the BART framework, to better measure which factors most influence end-task performance.')
    p4 = Document(metadata={'title':'Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer'}, page_content='Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts all text-based language problems into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new ``Colossal Clean Crawled Corpus'', we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our data set, pre-trained models, and code.')
    p5 = Document(metadata={'title':'Deep contextualized word representations'}, page_content='We introduce a new type of deep contextualized word representation that models both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy). Our word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pre-trained on a large text corpus. We show that these representations can be easily added to existing models and significantly improve the state of the art across six challenging NLP problems, including question answering, textual entailment and sentiment analysis. We also present an analysis showing that exposing the deep internals of the pre-trained network is crucial, allowing downstream models to mix different types of semi-supervision signals.')
    context = [p1,p2,p3,p4,p5]

    question = "RNN 이후에 LLM 분야의 시작을 알린 논문들을 추천해줘."

    answer = mock_llm_generate(question, context, OPENAI_API_KEY)

    print(answer)