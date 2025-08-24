#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DB 스키마 초기화 모듈 (PostgreSQL + pgvector)

- papers  : 논문 메타 + 초록 임베딩(벡터)
- citations: 인용 관계 (references 방향만 저장 가정)

스키마는 사용자가 제공한 표를 그대로 반영:
  - authors_json: TEXT (JSON 문자열)
  - embedding   : vector(768)  # pgvector
"""

from calendar import c
import os
import psycopg2
from psycopg2.extensions import connection as PGConnection
from pgvector.psycopg2 import register_vector


# -----------------------------
# 환경변수 (필요 시 .env 또는 러너 환경에 설정)
# -----------------------------
PGHOST = os.getenv("PGHOST", "/var/run/postgresql")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGUSER = os.getenv("PGUSER", "postgres")
PGPASSWORD = os.getenv("PGPASSWORD", "postgres")
PGDATABASE = os.getenv("PGDATABASE", "postgres")

# 임베딩 차원(스키마 고정값). 모델 바꾸면 여기만 수정.
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# 벡터 인덱스 lists 파라미터 (데이터가 많을수록 크게)
IVF_LISTS = int(os.getenv("PGVECTOR_IVF_LISTS", "100"))


# -----------------------------
# DDL (스키마 정의)
# -----------------------------
DDL_CREATE_EXTENSION = """
CREATE EXTENSION IF NOT EXISTS vector;
"""

DDL_TABLES = f"""
-- 논문 테이블
CREATE TABLE IF NOT EXISTS papers (
  arxiv_id        TEXT PRIMARY KEY,               -- 예: 1706.03762 (짧은 ID)
  openalex_id     TEXT,                           -- 예: https://openalex.org/W...
  doi             TEXT,
  title           TEXT NOT NULL,
  abstract        TEXT,
  authors_json    TEXT,                           -- JSON 문자열 (["A. Vaswani", ...])
  year            INTEGER,
  cited_by_count  INTEGER,
  embedding       VECTOR({EMBED_DIM}),            -- pgvector, 예: vector(768)
  created_at      TIMESTAMPTZ DEFAULT now(),
  updated_at      TIMESTAMPTZ DEFAULT now()
);

-- 인용 테이블(복합 PK): paper_openalex_id 가 참고한(references) related_work_id
CREATE TABLE IF NOT EXISTS citations (
  paper_openalex_id TEXT NOT NULL,
  related_work_id   TEXT NOT NULL,
  PRIMARY KEY (paper_openalex_id, related_work_id)
);
"""

DDL_UPDATED_AT_TRIGGER = """
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END; $$ LANGUAGE plpgsql;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname='papers_set_updated_at') THEN
    CREATE TRIGGER papers_set_updated_at
    BEFORE UPDATE ON papers
    FOR EACH ROW
    EXECUTE FUNCTION set_updated_at();
  END IF;
END $$;
"""

# -----------------------------
# 커넥션 & 초기화 함수
# -----------------------------
def get_conn() -> PGConnection:
    """
    PostgreSQL 커넥션 생성 + pgvector 어댑터 등록.
    """
    conn = psycopg2.connect(
        host=PGHOST, port=PGPORT, user=PGUSER, password=PGPASSWORD, dbname=PGDATABASE
    )
    return conn


def init_db(conn: PGConnection, *, with_ivf_index: bool = True) -> None:
    """
    DB 스키마 생성/보정
    - vector 확장
    - papers, citations 테이블
    - updated_at 트리거
    - 보조 인덱스
    - 벡터 IVFFlat 인덱스

    Parameters
    ----------
    conn : psycopg2 connection
        이미 열린 커넥션
    with_ivf_index : bool
        True면 IVFFlat 인덱스도 함께 생성 (기본값 True)
        초기 데이터가 거의 없을 땐 False로 시작 후 데이터 적재 뒤 생성/REINDEX 권장
    """
    with conn.cursor() as cur:
        cur.execute(DDL_CREATE_EXTENSION)
        cur.execute(DDL_TABLES)
        cur.execute(DDL_UPDATED_AT_TRIGGER)
        # 보조 인덱스 실행
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_papers_openalex_id ON papers(openalex_id);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_citations_paper ON citations(paper_openalex_id);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_citations_related ON citations(related_work_id);
        """)
        if with_ivf_index:
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_papers_embedding_ivf
                ON papers USING ivfflat (embedding vector_cosine_ops) WITH (lists = {IVF_LISTS});
            """)
        
    conn.commit()
    register_vector(conn)  # pgvector 컬럼에 파이썬 배열 바인딩 지원


# -----------------------------
# (선택) 유지보수 유틸
# -----------------------------
def drop_all(conn: PGConnection) -> None:
    """테스트용: 테이블과 인덱스만 삭제 (vector 확장은 유지)"""
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS citations CASCADE;")
        cur.execute("DROP TABLE IF EXISTS papers CASCADE;")
    conn.commit()


def reindex_vector(conn: PGConnection, lists: int) -> None:
    """
    데이터가 충분히 쌓인 뒤 벡터 인덱스 튜닝.
    """
    with conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS idx_papers_embedding_ivf;")
        cur.execute(f"""
            CREATE INDEX idx_papers_embedding_ivf
            ON papers USING ivfflat (embedding vector_l2_ops) WITH (lists = {lists});
        """)
    conn.commit()

if __name__ == "__main__":
    conn = get_conn()
    init_db(conn = conn, with_ivf_index=True)
    print("DB 초기화 완료")
    