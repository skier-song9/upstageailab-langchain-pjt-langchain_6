from concurrent.futures import thread
import uuid
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import json

from .graph.builder import build_graph
# LangGraph app 빌드
app_builder = build_graph()
# FastAPI app 생성
app = FastAPI(
    title='RAG API Server',
    description="LangGraph 기반 RAG 워크플로우 제어 서버"
)
class Phase1Request(BaseModel):
    query: str
class Phase2Request(BaseModel):
    thread_id: str
    question: str
    sbp_title: str

@app.post("start_phase1")
async def start_phase1(request: Phase1Request):
    """Phase 1 워크플로우를 시작하고, 논문 검색 결과와 함께 thread_id를 반환합니다.
    그래프는 select_paper 노드 이후 잠시 중단됩니다.

    :param Phase1Request request: input
    """
    # 새로운 대화를 위한 고유 ID 생성
    thread_id = str(uuid.uuid4())
    config = {"configurable":{"thread_id":thread_id}}
    inputs = {"initial_query":request.query, "thread_id": thread_id}

    # .stream()을 사용하여 그래프를 실행하고 중단점까지의 결과를 수집
    last_state = {}
    async for event in app_builder.astream(inputs, config):
        # select_paper 함수 이후의 state를 가져옴.
        last_state = event
    
    # state 에서 필요한 정보를 추출하여 클라이언트에 응답 > select_paper node 종료 이후의 GraphState를 가져온다.
    final_state = last_state.get('select_paper', {})

    return {
        'thread_id': thread_id,
        'sbp_found': final_state.get("sbp_found", False),
        'sbp_title': final_state.get('sbp_title', "Paper Not Found."),
    }

@app.post("start_phase2")
async def start_phase2(request: Phase2Request):
    """중단된 워크플로우를 이어받아 Phase 2를 실행하고, 최종 답변을 스트리밍으로 반환합니다.

    :param Phase2Request request: _description_
    """
    config = {"configurable": {"thread_id": request.thread_id}}
    async def stream_generator():
        # .stream()을 None 입력으로 호출하여 중단된 지점부터 실행 재개
        # 추가적인 입력을 state에 업데이트하여 전달
        async for event in app_builder.astream(
            {"question": request.question, "sbp_title": request.sbp_title}, config, stream_mode="values"
        ):
            if "generate_answer" in event:
                answer_chunk = event["generate_answer"]["answer"]
                # JSON 스트림 형식으로 데이터를 전송
                yield f"data: {json.dumps({'answer_chunk': answer_chunk})}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # uvicorn rag-api.src:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
