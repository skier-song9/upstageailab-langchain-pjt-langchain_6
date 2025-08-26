import os
import psycopg2
from psycopg2.extensions import connection as PGConnection
from pgvector.psycopg2 import register_vector
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT_DIR, "data")

# -----------------------------
# 환경변수 (필요 시 .env 또는 러너 환경에 설정)
# -----------------------------
PGHOST = os.getenv("PGHOST", "/var/run/postgresql")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGUSER = os.getenv("PGUSER", "postgres")
PGPASSWORD = os.getenv("PGPASSWORD", "postgres")
PGDATABASE = os.getenv("PGDATABASE", "postgres")

# 임베딩 차원(스키마 고정값). 모델 바꾸면 여기만 수정.
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))

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
  openalex_id     TEXT PRIMARY KEY,                           -- 예: W2896543
  doi             TEXT,
  title           TEXT NOT NULL,
  abstract        TEXT,
  authors         TEXT,                           -- 저자 이름 리스트 ("A. Vaswani", ...)
  pdf_url         TEXT,
  published       TIMESTAMP,
  cited_by_count  INTEGER,
  embedding       VECTOR({EMBED_DIM})            -- pgvector, 예: vector(1024)
);

-- 인용 테이블(복합 PK): citing_openalex_id 가 참고한(references) cited_openalex_id
CREATE TABLE IF NOT EXISTS citations (
  citing_openalex_id TEXT NOT NULL,
  cited_openalex_id   TEXT NOT NULL,
  PRIMARY KEY (citing_openalex_id, cited_openalex_id)
);
"""

# DDL_UPDATED_AT_TRIGGER = """
# CREATE OR REPLACE FUNCTION set_updated_at()
# RETURNS TRIGGER AS $$
# BEGIN
#   NEW.updated_at = now();
#   RETURN NEW;
# END; $$ LANGUAGE plpgsql;

# DO $$
# BEGIN
#   IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname='papers_set_updated_at') THEN
#     CREATE TRIGGER papers_set_updated_at
#     BEFORE UPDATE ON papers
#     FOR EACH ROW
#     EXECUTE FUNCTION set_updated_at();
#   END IF;
# END $$;
# """

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
        # cur.execute(DDL_UPDATED_AT_TRIGGER)
        # 보조 인덱스 실행
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_papers_openalex_id ON papers(openalex_id);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_citations_paper ON citations(citing_openalex_id);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_citations_related ON citations(cited_openalex_id);
        """)
        if with_ivf_index:
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_papers_embedding_ivf
                ON papers USING ivfflat (embedding vector_cosine_ops) WITH (lists = {IVF_LISTS});
            """)
        
    conn.commit()
    register_vector(conn)  # pgvector 컬럼에 파이썬 배열 바인딩 지원

def insert_papers(conn: PGConnection, meta_df: pd.DataFrame, emb_npy: np.ndarray) -> None:
    """
    논문 메타데이터를 테이블에 삽입
    """
    with conn.cursor() as cur:
        for idx, row in meta_df.iterrows():
            embedding = emb_npy[idx].tolist()
            cur.execute("""
                INSERT INTO papers (openalex_id, doi, title, abstract, authors, pdf_url, published, cited_by_count, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (row["openalex_id"], row["doi"], row["title"], row["abstract"], row["authors"], row["pdf_url"], row["publication_date"], row["cited_by_count"], embedding))
    
def insert_citations(conn: PGConnection, citations_df: pd.DataFrame) -> None:
    """
    인용 관계를 테이블에 삽입
    """
    with conn.cursor() as cur:
        for idx, row in citations_df.iterrows():
            cur.execute("""
                INSERT INTO citations (citing_openalex_id, cited_openalex_id)
                VALUES (%s, %s)
            """, (row["citing_paper_id"], row["cited_paper_id"]))
    conn.commit()

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

    meta_df = pd.read_csv(os.path.join(data_dir, "papers.csv"))
    emb_npy = np.load(os.path.join(data_dir, "papers_embeddings.npy"))
    insert_papers(conn, meta_df, emb_npy)
    print("논문 삽입 완료")
    
    citations_df = pd.read_csv(os.path.join(data_dir, "citations.csv"))
    insert_citations(conn, citations_df)
    print("인용 관계 삽입 완료")

    