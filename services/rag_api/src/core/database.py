# def mock_db_select(paper_title: str) -> dict | None:
#     """DB에서 논문을 조회하는 함수. 기본적으로 "논문의 제목"을 바탕으로 일치하는 논문을 찾는 로직으로 구현한다.
#     (사용자가 읽은 논문이니 full title을 입력해줄 것으로 기대)

#     1. DB에서 일치하는 논문 또는 유사한 제목의 논문을 찾아 반환한다.
    
#     :param paper_title str:
#     """
#     print(f"📄 DB 조회: '{paper_title}'")
    
#     if "graph rag" in paper_title.lower():
#         return {"title": "Graph RAG", "is_sbp": True, "details": "Graph RAG에 대한 상세 정보"}
#     return None

# def mock_db_insert(paper_info: dict):
#     """DB에 논문 정보를 삽입하는 모의 함수."""
#     print(f"💾 DB에 삽입: '{paper_info['title']}'")
#     # 실제로는 DB에 저장하는 로직이 들어갑니다.
#     pass

# def mock_db_follow_up_select(paper_title: str) -> list[str]:
#     """DB에서 인용 관계의 후속 논문을 조회하는 모의 함수."""
#     print(f"🔍 DB 인용관계 검색 (Select): '{paper_title}' 인용 논문")
#     return ["후속 논문 C (from DB)", "후속 논문 D (from DB)"]

from db.db_init import get_conn

import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from pgvector.psycopg2 import register_vector


def mock_db_select(paper_title: str) -> dict | None:
    print(f"📄 DB 조회: '{paper_title}'")

    conn = get_conn()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT title
                FROM papers
                WHERE title = %s
                LIMIT 1
            """, (paper_title,))
            row = cur.fetchone()
            if row:
                return {
                    "title": row["title"],
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

    authors = ", ".join([a['author']['display_name'] for a in paper_info.get("authorships", [])])

    try:
        with conn.cursor() as cur:
            # papers 테이블에 삽입
            cur.execute("""
                INSERT INTO papers (
                    openalex_id, title, publication_date, doi, cited_by_count, abstract, pdf_url, authors, embedding
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
                authors,                    # 입력 형태에 따라 수정 필요
                paper_info.get("embedding") # 입력 형태에 따라 수정 필요
            ))

            # citations 테이블에 삽입
            citing_paper_id = paper_info.get("openalex_id")
            cited_papers = paper_info.get("cited_papers", []) # 입력 형태에 따라 수정 필요

            # (citing, cited) 튜플 리스트 생성
            rows = [(citing_paper_id, cited_id) for cited_id in cited_papers]

            if rows:
                execute_values(cur, """
                    INSERT INTO citations (paper_openalex_id, related_work_id)
                    VALUES %s
                    ON CONFLICT DO NOTHING
                """, rows)

        conn.commit()
    finally:
        conn.close()

def mock_db_follow_up_select(paper_title: str) -> list[str]:
    print(f"🔍 DB 인용관계 검색 (Select): '{paper_title}' 인용 논문")
    
    conn = get_conn()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # title로 openalex_id 찾기
            cur.execute("""
                SELECT openalex_id
                FROM papers
                WHERE title = %s
                LIMIT 1
            """, (paper_title,))
            row = cur.fetchone()
            if not row:
                return []  # 논문이 없으면 빈 리스트 반환

            target_openalex_id = row["openalex_id"]

            # citations 테이블에서 cited_openalex_id가 일치하는 모든 citing_openalex_id 조회
            cur.execute("""
                SELECT citing_openalex_id
                FROM citations
                WHERE cited_openalex_id = %s
            """, (target_openalex_id,))

            results = cur.fetchall()
            # 리스트로 변환
            return [r["citing_openalex_id"] for r in results]

    finally:
        conn.close()
