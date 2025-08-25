from dataclasses import asdict, dataclass
from typing import Optional, List, Dict
import pandas as pd
import time
from datetime import datetime
import requests
import feedparser
from requests.exceptions import ReadTimeout, ConnectionError, HTTPError
import random

@dataclass
class PaperMeta:
    arxiv_id: str
    title: str
    abstract: str
    authors: [str]                 # "Lastname, Firstname; ..." 형태로 직렬화
    primary_category: str
    categories: str              # "cs.CL; stat.ML" 형태
    published: str               # ISO8601
    pdf_url: str
    journal_ref: str

def _to_meta(r) -> PaperMeta:
    authors = r["authors"]
    primary_category = r["primary_category"]
    cats = r["categories"]
    pdf_url = r["pdf_url"]
    return PaperMeta(
        arxiv_id=r["arxiv_id"],
        title=r["title"],
        abstract=r["abstract"],
        authors=authors or "",
        primary_category=primary_category or "",
        categories=cats or "",
        published=r["published"] or "",
        pdf_url=pdf_url or "",
        journal_ref=r["journal_ref"] or "",
    )


def build_query(
    keywords: List[str],
    fields: str = "all",   # "all" | "title" | "abstract" | "ti,abs"
    extra: Optional[str] = None
) -> str:
    """
    arXiv 검색 문법:
      - ti:"transformer" AND abs:"speech"
      - cat:cs.CL
      - submittedDate:[2024-01-01 TO 2025-08-25]
    """
    def one_kw(k: str) -> str:
        k = k.strip()
        if fields == "title":
            return f'ti:{k}'
        elif fields == "abstract":
            return f'abs:{k}'
        elif fields == "ti,abs":
            return f'ti:{k} OR abs:{k}'
        else:
            return f'all: {k}'

    cores = [one_kw(k) for k in keywords if k.strip()]
    if extra:
        cores = [f"{core} AND {extra}" for core in cores]
    return cores

def fetch_page(search_query: str, start: int, batch: int,
               sortBy: str = "relevance", sortOrder: str = "descending",
               timeout=(5, 60),  # (connect, read)
               sleep_jitter: tuple = (0.4, 1.0),
               ignore_timeout: bool = True):
    '''
    search_query: arxiv 검색 시 사용하는 쿼리 (ex: ti:attention OR abs:attention AND (cat:cs.CV OR cat:cs.LG OR cat:cs.AI OR cat:stat.ML OR cat:cs.CL OR cat:cs.MA) )
    start: 페이지 시작 인덱스
    batch: 페이지 당 결과 개수
    sortBy: 정렬 기준 (relevance, lastUpdatedDate, submittedDate)
    sortOrder: 정렬 순서 (ascending, descending)
    timeout: 타임아웃 시간 (connect, read)
    sleep_jitter: 조용히 스킬링 완화를 위한 지연 시간 (0.4, 1.0)
    ignore_timeout: 타임아웃 시 조용히 스킬링 완화를 위한 플래그
    '''

    params = {
        "search_query": search_query,
        "start": start,
        "max_results": batch,  # arXiv는 보통 300이 상한, 100~200 권장
        "sortBy": sortBy,
        "sortOrder": sortOrder,
    }

    SESSION = requests.Session()
    URL = "https://export.arxiv.org/api/query"
    HEADERS = {
        "User-Agent": "arxiv-harvester/0.1 (wjdqlsrla0309@naver.com)",
        "From": "wjdqlsrla0309@naver.com",
        "Accept-Encoding": "gzip, deflate",
}

    # 예의상 약간의 지연(스로틀링 완화)
    time.sleep(random.uniform(*sleep_jitter))

    try:
        r = SESSION.get(URL, params=params, timeout=timeout, headers=HEADERS)
        r.raise_for_status()
    except (ReadTimeout, ConnectionError) as e:
        # 타임아웃/네트워크 실패는 조용히 스킵 (로그만)
        if ignore_timeout:
            print(f"[WARN] fetch_page timeout/conn error at start={start}, batch={batch}: {e}. Skipping this page.")
            return []  # 빈 피드로 간주하고 다음 루프로 진행
        else:
            raise
    except HTTPError as e:
        # 5xx는 한 번 정도 느슨하게 재시도해도 좋음 (선택)
        if 500 <= getattr(e.response, "status_code", 0) < 600 and ignore_timeout:
            try:
                time.sleep(1.2)
                r = SESSION.get(URL, params=params, timeout=(5, 90), headers=HEADERS)
                r.raise_for_status()
            except Exception as e2:
                print(f"[WARN] fetch_page HTTP {getattr(e.response, 'status_code', '5xx')} again failed: {e2}. Skipping.")
                return []
        else:
            # 4xx는 보통 쿼리 문제 -> 바로 전파
            raise

    feed = feedparser.parse(r.text)
    total = int(getattr(feed.feed, "opensearch_totalresults", 0))
    per = int(getattr(feed.feed, "opensearch_itemsperpage", 0))
    start_idx = int(getattr(feed.feed, "opensearch_startindex", 0))
    print(f"total: {total}, per: {per}, start_idx: {start_idx}")

    return feed.entries  # list[entry]

def paged_arxiv(search_query: str, total: int = 1000, batch: int = 20,
                sortBy: str = "relevance", sortOrder: str = "descending", sleep_time: int = 4):

    '''
    search_query: arxiv 검색 시 사용하는 쿼리 (ex: ti:attention OR abs:attention AND (cat:cs.CV OR cat:cs.LG OR cat:cs.AI OR cat:stat.ML OR cat:cs.CL OR cat:cs.MA) )
    total: 총 결과 개수
    batch: 페이지 당 결과 개수
    sortBy: 정렬 기준 (relevance, lastUpdatedDate, submittedDate)
    sortOrder: 정렬 순서 (ascending, descending)
    sleep_time: 페이지 간 대기 시간
    '''
    seen = set()
    for start in range(0, total, batch):
        
        for i in range(5):
            
            entries = fetch_page(search_query, max(0, start-i), batch+i, sortBy, sortOrder)
            print(f"len entries: {len(entries)}")
            if len(entries) >= batch:
                break
            time.sleep(sleep_time)
        
        if not entries:
            continue
        
        for e in entries:
            aid = e.id.split("/")[-1]
            if aid in seen: 
                continue
            seen.add(aid)
            yield {
                "arxiv_id": aid,
                "title": e.title,
                "abstract": getattr(e, "summary", ""),
                "published": getattr(e, "published", ""),
                "authors": "; ".join(a["name"] for a in e.authors),
                "primary_category": getattr(e, "arxiv_primary_category", {}).get("term", ""),
                "categories": "; ".join(t["term"] for t in e.tags),
                "pdf_url": next((l["href"] for l in e.links if l.get("type") == "application/pdf"), None),
                "journal_ref": getattr(e, "arxiv_journal_ref", ""),
            }
        time.sleep(sleep_time)  # arXiv 권장 대기 (페이지 간 3초). :contentReference[oaicite:1]{index=1}

def fetch_arxiv_metadata(
    keywords: List[str],
    fields: str = "all",
    max_results: int = 200,
    batch: int = 500,
    sort: str = "relevance",   # "relevance" | "lastUpdatedDate" | "submittedDate"
    sort_order: str = "descending",  # "ascending" | "descending"
    date_from: Optional[str] = None, # "YYYY-MM-DD"
    date_to: Optional[str] = None,   # "YYYY-MM-DD"
    category: Optional[str] = None,  # 예: "cs.CL" or "cs.LG"
    extra: Optional[str] = None,
) -> pd.DataFrame:
    '''
    keywords: 검색 키워드
    fields: 검색 필드 (all, title, abstract, ti,abs)
    max_results: 총 결과 개수
    batch: 페이지 당 결과 개수
    sort: 정렬 기준 (relevance, lastUpdatedDate, submittedDate)
    sort_order: 정렬 순서 (ascending, descending)
    date_from: 날짜 범위 시작
    date_to: 날짜 범위 종료
    category: 카테고리
    extra: 추가 질의
    '''
    # 날짜/카테고리 필터 추가
    extras = []
    if category:
        categories = " OR ".join([f"cat:{cat}" for cat in category])
        extras.append(f"({categories})")
    if date_from or date_to:
        # submittedDate 필터
        fr = date_from or "19000101"
        to = date_to or datetime.utcnow().date().strftime("%Y%m%d")
        extras.append(f"(submittedDate:[{fr} TO {to}])")
    if extra:
        extras.append(extra)
    extra_str = " AND ".join(extras) if extras else None

    querys = build_query(keywords, fields=fields, extra=extra_str)


    results = []
    for query in querys:
        print(f"query: {query}")
        result = list(paged_arxiv(query, sortBy=sort, sortOrder=sort_order, total=max_results, batch=batch))
        results.extend(result)

    rows: List[Dict] = []
    seen = set()
    count = 0
    for r in results:
        meta = _to_meta(r)
        if meta.arxiv_id in seen:
            continue
        seen.add(meta.arxiv_id)
        rows.append(asdict(meta))
        count += 1

    print(f"{count} rows")
    df = pd.DataFrame(rows)
    if not df.empty:
        # 정렬 통일(최신 업데이트 우선)
        df.sort_values(["published"], ascending=[False], inplace=True)
        # 가끔 공백/개행 정리
        df["title"] = df["title"].str.replace(r"\s+", " ", regex=True).str.strip()
        df["abstract"] = df["abstract"].str.replace(r"\s+", " ", regex=True).str.strip()
    return df


if __name__ == "__main__":  
    keywords = ["transformer", "diffusion", "language", "tuning", "retrieval", "neural", "attention"]
    category = ["cs.CV", "cs.LG", "cs.AI", "stat.ML", "cs.CL", "cs.MA"]
    
    df = fetch_arxiv_metadata(
        keywords=keywords,
        fields="ti,abs",           
        max_results=2400,
        batch=400,
        sort="relevance",
        sort_order="descending",
        date_from=None,
        date_to=None,
        category=category,                  # 예: "cs.CL" 또는 "cs.CV"
        extra=None                      # 추가 질의가 필요하면 여기에 arXiv 문법으로
    )

    if not df.empty:
        df.to_csv("arxiv_meta.csv", index=False)