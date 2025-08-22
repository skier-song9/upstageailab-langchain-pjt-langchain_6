import os
import sys

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.graph.builder import build_graph

# --- 환경 설정 (필요 시) ---
# os.environ["LANGCHAIN_API_KEY"] = "YOUR_API_KEY"
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# 그래프 빌드
app = build_graph()

# --- 그래프 시각화 ---
# Mermaid 다이어그램 생성
try:
    mermaid_png = app.get_graph().draw_mermaid_png()
    
    # outputs 디렉토리 경로 설정
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # PNG 파일로 저장
    with open(os.path.join(output_dir, "graph_workflow.png"), "wb") as f:
        f.write(mermaid_png)
    print(f"✅ 그래프 시각화 완료: {os.path.join(output_dir, 'graph_workflow.png')}")

except Exception as e:
    print(f"❌ 그래프 시각화 실패: {e}")
    print("ℹ️ 'graphviz'와 'pygraphviz'가 설치되어 있는지 확인해주세요.")


# --- 그래프 실행 (테스트용) ---
print("\n" + "="*30)
print("🚀 그래프 실행 테스트")
print("="*30)

# 시나리오 1: DB에 SBP가 바로 없는 경우
print("\n===== 시나리오 1: 'Some other paper' 검색 시작 =====")
inputs1 = {"initial_query": "Some other paper"}
for event in app.stream(inputs1):
    for key, value in event.items():
        print(f"--- 실행된 노드: {key} ---")
        print(value)
print("\n최종 결과 (시나리오 1):", app.invoke(inputs1)["answer"])


# 시나리오 2: DB에 SBP가 바로 있는 경우
print("\n\n===== 시나리오 2: 'Graph RAG' 검색 시작 =====")
inputs2 = {"initial_query": "Graph RAG"}
final_state = app.invoke(inputs2)
print("\n최종 결과 (시나리오 2):", final_state["answer"])
