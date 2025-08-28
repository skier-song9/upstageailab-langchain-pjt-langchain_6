import os
import sys
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector

ROOT_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", "..", "..", ".."))
sys.path.append(ROOT_DIR)

from services.rag_api.src.core.source_api import openalex_search
from services.rag_api.src.core.get_emb import get_emb_model, get_emb
from db.db_init import get_conn

def mock_db_select(paper_title: str) -> dict | None:
    print(f"📄 DB 조회: '{paper_title}'")

    try:
        conn = get_conn()
        register_vector(conn)

        print(f"DB 연결 성공")
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        return None

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT *
                FROM papers
                WHERE title ilike %s
                LIMIT 1
            """, (paper_title,))
            row = cur.fetchone()
            if row:
                return {
                    "paper_meta": row,
                    "is_sbp": True
                }
            else:
                return None
    finally:
        conn.close()

def mock_db_insert(paper_info: dict):
    print(f"💾 DB에 삽입: '{paper_info}'")

    conn = get_conn()
    register_vector(conn)

    try:
        with conn.cursor() as cur:
            # papers 테이블에 삽입
            cur.execute("""
                INSERT INTO papers (
                    openalex_id, title, published, doi, cited_by_count, abstract, pdf_url, authors, embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (openalex_id) DO NOTHING
            """, (
                paper_info.get("openalex_id"),
                paper_info.get("title"),
                paper_info.get("publication_date"),
                paper_info.get("doi"),
                paper_info.get("cited_by_count"),
                paper_info.get("abstract"),
                paper_info.get("pdf_url"),
                paper_info.get("authors"),                    # 입력 형태에 따라 수정 필요
                paper_info.get("embedding") # 입력 형태에 따라 수정 필요
            ))

            # citations 테이블에 삽입
            citing_paper_id = paper_info.get("openalex_id")
            cited_papers = paper_info.get("cited_papers", []) # 입력 형태에 따라 수정 필요

            # (citing, cited) 튜플 리스트 생성
            rows = [(citing_paper_id, cited_id) for cited_id in cited_papers]

            if rows:
                execute_values(cur, """
                    INSERT INTO citations (citing_openalex_id, cited_openalex_id)
                    VALUES %s
                    ON CONFLICT DO NOTHING
                """, rows)

        conn.commit()
    finally:
        conn.close()

def mock_db_follow_up_select(paper_info: dict, query_vec: list[float], k: int) -> list[str]:
    print(f"🔍 DB 인용관계 검색 (Select): '{paper_info['title']}' 인용 논문")
    
    conn = get_conn()
    register_vector(conn)

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:

            cur.execute("""
                SELECT citing_openalex_id
                FROM citations
                WHERE cited_openalex_id = %s
            """, (paper_info["openalex_id"],))

            results = cur.fetchall()
            ids = [r['citing_openalex_id'] for r in results]

            if not ids:
                return []

            cur.execute("""
                        WITH RankedPapers AS (
                            SELECT
                                p.openalex_id,
                                p.title,
                                p.published,
                                p.abstract,
                                p.pdf_url,
                                p.authors,
                                p.cited_by_count,
                                p.embedding <=> %s AS dist,
                                ROW_NUMBER() OVER(PARTITION BY p.title ORDER BY LENGTH(p.abstract) DESC) AS rn
                            FROM papers p
                            JOIN citations c ON c.citing_openalex_id = p.openalex_id
                            WHERE c.cited_openalex_id = ANY(%s)
                        )
                        SELECT
                            openalex_id,
                            title,
                            published,
                            abstract,
                            pdf_url,
                            authors,
                            cited_by_count,
                            dist
                        FROM RankedPapers
                        WHERE rn = 1
                        ORDER BY dist
                        LIMIT %s
                        """, (query_vec, ids, k))

            rows = cur.fetchall()
            return rows

    finally:
        conn.close()

if __name__ == "__main__":
    paper_info = openalex_search("attention is all you need")
    print(paper_info)

    query = "Tell me about a paper that improved the computational efficiency of the attention mechanism."
    emb_model = get_emb_model()
    query_vec = get_emb(emb_model, [query])
    print(query_vec)
    print(mock_db_follow_up_select(paper_info, query_vec[0], 5))