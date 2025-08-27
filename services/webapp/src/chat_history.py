from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_upstage import ChatUpstage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
os.environ["UPSTAGE_API_KEY"] = "up_m3tvzbIfxJ7YMDPeaiYEvsuw7XNew"

model = ChatUpstage(
    model="solar-mini", 
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 내가 지정한 논문의 후속 논문을 추천해주는 연구 도우미입니다.",
        ),
        # 대화 기록을 변수로 사용, history 가 MessageHistory 의 key 가 됨
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),  # 사용자 입력을 변수로 사용
    ]
)
runnable = prompt | model  # 프롬프트와 모델을 연결하여 runnable 객체 생성

store = {}  # 세션 기록을 저장할 딕셔너리

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    print(session_ids)
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


with_message_history = (
    RunnableWithMessageHistory(  # RunnableWithMessageHistory 객체 생성
        runnable,  # 실행할 Runnable 객체
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="input",  # 입력 메시지의 키
        history_messages_key="history",  # 기록 메시지의 키
    )
)

response = with_message_history.invoke(
    # 수학 관련 질문 "코사인의 의미는 무엇인가요?"를 입력으로 전달합니다.
    {"input": "'Learning Deep Architectures for AI' 논문의 후속 논문을 추천해줘."},
    # 설정 정보로 세션 ID "QA"을 전달합니다.
    config={"configurable": {"session_id": "QA"}},
)

print(response)

response = with_message_history.invoke(
    # 수학 관련 질문 "코사인의 의미는 무엇인가요?"를 입력으로 전달합니다.
    {"input": "네가 말해줬던 첫번째 논문의 제목만 다시 말해줘."},
    # 설정 정보로 세션 ID "QA"을 전달합니다.
    config={"configurable": {"session_id": "QA"}},
)

print(response)
#출력 결과
#QA
#AIMessage(content='Cosine is a trigonometric function that represents the ratio of the adjacent side to the hypotenuse in a right triangle.', response_metadata={'token_usage': {'completion_tokens': 26, 'prompt_tokens': 47, 'total_tokens': 73}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_b28b39ffa8', 'finish_reason': 'stop', 'logprobs': None})

# with_message_history.invoke(
#     # 수학 관련 질문 "코사인의 의미는 무엇인가요?"를 입력으로 전달합니다.
#     {"input": "추천한 후속 논문"},
#     # 설정 정보로 세션 ID "QA"을 전달합니다.
#     config={"configurable": {"session_id": "QA"}},
# )