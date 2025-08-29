## Porject Directory Structure
upstageailab-langchain-pjt-langchain_6/
├── services/
│   ├── paper-ingestor/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── src/
│   │       ├── __main__.py
│   │       ├── config.py
│   │       ├── database.py
│   │       ├── embedding.py
│   │       ├── scheduler.py
│   │       └── source_api.py
│   │
│   └── rag-api/
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── visualize.py
│       └── src/
│           ├── __main__.py
│           ├── api/
│           │   ├── __init__.py
│           │   ├── endpoints.py
│           │   └── schemas.py
│           ├── config.py
│           ├── graph/
│           │   ├── __init__.py
│           │   ├── builder.py
│           │   ├── nodes.py
│           │   └── state.py
│           └── core/
│               ├── __init__.py
│               ├── database.py
│               ├── get_emb.py
│               ├── llm.py
│               ├── retriever.py
│               └── source_api.py
│
├── packages/
│   └── common/
│       ├── __init__.py
│       └── models.py
│
├── webapp
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── __init__.py
│       ├── app.py
│       └── chat_history.py
│
├── .env
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── README.md

#### Descriptions
- Root Level
    | 파일/디렉토리 |	역할 / 기능 |
    | --- | --- |
    | services/ |	각 마이크로서비스 애플리케케이션의 코드가 위치하는 메인 디렉토리입니다. |
    |packages/ |	여러 서비스에서 공통으로 사용하는 코드(예: DB 모델)를 관리하는 디렉토리입니다. |
    |.env |	DB 접속 정보, LLM API 키 등 민감한 환경 변수를 관리하는 파일입니다. |
    |.gitignore |	Git 버전 관리에서 제외할 파일 및 디렉토리 목록을 정의합니다. |
    |docker-compose.yml |	로컬 개발 환경에서 여러 서비스(ingestor, api, postgres)를 한번에 실행하고 관리합니다. |
    |README.md |	프로젝트 설치, 실행 방법, 아키텍처 등 전반적인 내용을 설명하는 문서입니다. |

- services/paper-ingestor/ (Phase 0: 데이터 수집 서비스)
    | 파일/디렉토리	|역할 / 기능 |
    | --- | --- |
    | Dockerfile|	paper-ingestor 서비스를 컨테이너 이미지로 빌드하기 위한 명세서입니다. |
    | requirements.txt|	이 서비스에 필요한 Python 라이브러리 목록입니다. (e.g., requests, psycopg2, sentence-transformers)
    | src/__main__.py|	서비스의 시작점(entrypoint)으로, 스케줄러를 실행시키는 코드가 위치합니다. |
    | src/config.py|	.env 파일로부터 환경 변수를 읽어와 설정 객체를 생성합니다. |
    | src/database.py|	PostgreSQL DB에 연결하고 논문 데이터 및 임베딩을 저장/업데이트하는 함수를 관리합니다. |
    | src/embedding.py|	Hugging Face 등에서 가져온 오픈 임베딩 모델을 로드하고 텍스트를 벡터로 변환합니다. |
    | src/scheduler.py|	APScheduler 등을 사용하여 주기적으로 논문 수집 및 임베딩 작업을 실행합니다. |
    | src/source_api.py|	OpenAlex API 등 외부 소스로부터 논문 데이터를 가져오는 클라이언트 코드입니다. |

- services/rag-api/ (Phase 1 & 2: RAG API 서비스)
    | 파일/디렉토리 | 역할 / 기능 |
    | :--- | :--- |
    | `Dockerfile` | rag-api 서비스를 컨테이너 이미지로 빌드하기 위한 명세서입니다. |
    | `requirements.txt` | 이 서비스에 필요한 Python 라이브러리 목록입니다. (e.g., fastapi, uvicorn, langchain, langgraph) |
    | `visualize.py` | 그래프의 실행을 테스트하고, `outputs/` 디렉토리에 워크플로우를 시각화한 이미지를 저장합니다. |
    | `src/__main__.py` | 서비스의 시작점(entrypoint)으로, FastAPI 애플리케케이션을 실행시키는 코드가 위치합니다. |
    | `src/api/` | FastAPI를 사용한 API 엔드포인트 관련 코드를 관리합니다. |
    | `src/api/endpoints.py` | RAG 파이프라인을 호출하는 API 엔드포인트를 정의합니다. |
    | `src/api/schemas.py` | API 요청/응답에 사용되는 데이터 모델(Pydantic 모델)을 정의합니다. |
    | `src/config.py` | .env 파일로부터 환경 변수를 읽어와 설정 객체를 생성합니다. |
    | `src/graph/` | LangGraph를 사용하여 RAG 워크플로우를 정의하고 관리합니다. |
    | `src/graph/state.py` | 그래프의 각 노드를 거치며 데이터가 저장되고 업데이트되는 상태 객체(`GraphState`)를 정의합니다. |
    | `src/graph/nodes.py` | 그래프를 구성하는 각 노드(Node)의 실제 실행 함수들을 정의합니다. |
    | `src/graph/builder.py` | `state.py`와 `nodes.py`를 바탕으로 LangGraph 워크플로우를 구성하고 컴파일합니다. |
    | `src/core/` | RAG의 핵심 로직(DB, LLM, Retriever)을 담당하는 모듈을 관리합니다. |
    | `src/core/database.py` | DB에서 데이터를 조회하거나 저장하는 함수들을 관리합니다. |
    | `src/core/llm.py` | LLM을 호출하여 답변을 생성하는 함수를 관리합니다. |
    | `src/core/retriever.py` | Vector Store에서 관련 문서를 검색하는 Retriever 함수를 관리합니다. |
    | `src/core/source_api.py` | 외부 소스(웹 검색 등)로부터 데이터를 가져오는 함수를 관리합니다. |

- packages/common/ (공통 패키지)
    - models.py :	SQLAlchemy 등을 사용하여 PostgreSQL 테이블과 매핑되는 공통 데이터 모델(ORM 모델)을 정의합니다.
