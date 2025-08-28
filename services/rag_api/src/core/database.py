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
    """
    논문 제목을 기반으로 데이터베이스에서 논문을 검색합니다.
    대소문자를 구분하지 않는 'ilike'를 사용하여 부분 일치 검색을 수행합니다.

    :param paper_title: 검색할 논문 제목 문자열
    :return: 검색된 논문 정보(메타데이터)와 검색 성공 여부를 담은 딕셔너리. 찾지 못하면 None을 반환합니다.
    """
    print(f"📄 DB 조회: '{paper_title}'")

    try:
        conn = get_conn() # 데이터베이스 연결 가져오기
        register_vector(conn) # pgvector 사용을 위해 벡터 타입 등록

        print(f"DB 연결 성공")
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        return None

    try:
        # RealDictCursor를 사용하여 결과를 딕셔너리 형태로 받음
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # papers 테이블에서 제목이 일치하는 논문을 1개 검색
            cur.execute("""
                SELECT *
                FROM papers
                WHERE title ilike %s
                LIMIT 1
            """, (f"%{paper_title}%",)) # 양쪽에 %를 추가하여 부분 일치 검색
            row = cur.fetchone()
            if row: # 논문을 찾았다면 결과 반환
                return {
                    "paper_meta": row,
                    "is_sbp": True
                }
            else: # 찾지 못했다면 None 반환
                return None
    finally:
        conn.close() # DB 연결 종료

def mock_db_insert(paper_info: dict):
    """
    OpenAlex에서 검색한 논문 정보(메타데이터, 인용 관계)를 데이터베이스에 삽입합니다.

    :param paper_info: 저장할 논문 정보가 담긴 딕셔너리
    """
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
    """
    주어진 기준 논문(paper_info)을 인용한 후속 연구들을 검색하고,
    사용자의 질문 벡터(query_vec)와 가장 유사한 상위 k개의 논문을 반환합니다.

    :param paper_info: 기준이 되는 논문의 정보 딕셔너리 (openalex_id 포함)
    :param query_vec: 사용자의 질문을 임베딩한 벡터
    :param k: 가져올 후속 논문의 최대 개수
    :return: 검색된 후속 논문 정보 딕셔너리의 리스트
    """
    print(f"🔍 DB 인용관계 검색 (Select): '{paper_info['title']}' 인용 논문")
    
    conn = get_conn()
    register_vector(conn)

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1. 기준 논문(paper_info)을 인용한 논문들의 ID(citing_openalex_id)를 조회
            cur.execute("""
                SELECT citing_openalex_id
                FROM citations
                WHERE cited_openalex_id = %s
            """, (paper_info["openalex_id"],))

            results = cur.fetchall()
            ids = [r['citing_openalex_id'] for r in results] # 조회된 ID들을 리스트로 변환

            if not ids: # 후속 연구가 없으면 빈 리스트 반환
                return []

            # 2. 후속 연구 ID 리스트를 사용하여, 해당 논문들 중에서 사용자 질문 벡터와 코사인 유사도가 가장 높은 상위 k개를 검색
            # (<=> 연산자는 pgvector에서 코사인 거리를 계산함)
            cur.execute("""
                SELECT p.openalex_id, p.title, p.published, p.abstract, p.embedding <=> %s AS dist
                FROM papers p
                JOIN citations c ON c.citing_openalex_id = p.openalex_id
                WHERE c.cited_openalex_id = ANY(%s)
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