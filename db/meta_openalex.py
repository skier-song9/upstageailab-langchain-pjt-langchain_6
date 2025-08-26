from util import get_json, OPENALEX, PER_PAGE, RPS_SLEEP, reconstruct_abstract
import time
import pandas as pd
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_dir = os.path.join(ROOT_DIR, "data")

def search_works_by_keywords(
    query: str,
    filters: dict | None = None,
    max_records: int = 5000,
    use_cursor: bool = True,
    select_fields: str = "id,display_name,publication_year,doi,cited_by_count,abstract_inverted_index,authorships,primary_location,referenced_works",
    per_page: int = PER_PAGE,
    rps_sleep: float = RPS_SLEEP,
):
    """
    query: OpenAlex search= (전체 텍스트 검색) 쿼리 문자열
    filters: {"publication_year": "2020|2021", "type": "journal-article", ...}
    max_records: 최대 수집 개수
    use_cursor: True면 cursor 페이징, False면 page 기반
    """
    params = {"search": query, "per-page": per_page, "select": select_fields}
    if filters:
        # ex) "publication_year:2020|2021,type:journal-article"
        filt = ",".join(f"{k}:{v}" for k, v in filters.items() if v)
        if filt:
            params["filter"] = filt

    items, got = [], 0
    if use_cursor:
        params["cursor"] = "*"
        while True:
            j = get_json(f"{OPENALEX}/works", params)
            results = j.get("results", [])
            for w in results:
                items.append(w)
                got += 1
                if got >= max_records:
                    return items
            nxt = j.get("meta", {}).get("next_cursor")
            print(f"query: {query}, items: {len(items)}")
            if not nxt: break
            params["cursor"] = nxt
            time.sleep(rps_sleep)
    else:
        page = 1
        while got < max_records:
            params["page"] = page
            j = get_json(f"{OPENALEX}/works", params)
            results = j.get("results", [])
            if not results: break
            items.extend(results)
            got += len(results)
            page += 1
            time.sleep(rps_sleep)
    return items

def harvest_openalex_by_keywords(
    keywords: list[str],
    filters: dict | None = None,
    max_records: int = 2000
):
    works = []
    for keyword in keywords:
        # 메타 수집
        work = search_works_by_keywords(
            query=keyword,
            filters=filters,
            max_records=max_records,
            use_cursor=True,
            select_fields="id,display_name,publication_date,doi,cited_by_count,abstract_inverted_index,primary_location,authorships,referenced_works"
        )
        works.extend(work)
    # papers 테이블
    papers_rows = []
    # 참고 논문 테이블
    citation_rows = []
    
    seen_papers = set()
    for w in works:
        if w["id"] in seen_papers:
            continue

        abs_txt = reconstruct_abstract(w.get("abstract_inverted_index"))
        if not abs_txt.strip():
            continue

        if not w.get("authorships"):
            continue

        authors = ", ".join([a['author']['display_name'] for a in w.get("authorships", [])])

        papers_rows.append({
            "openalex_id": w["id"].split("/")[-1],
            "title": w.get("display_name"),
            "publication_date": w.get("publication_date"),
            "doi": w.get("doi"),
            "cited_by_count": w.get("cited_by_count", ""),
            "abstract": abs_txt,
            "pdf_url": (w.get("primary_location") or {}).get("pdf_url"),
            "authors": authors,
        })

        for ref in w.get("referenced_works", []):
            if w["id"] == ref:
              continue
            citation_rows.append({
                "citing_paper_id": w["id"].split("/")[-1],
                "cited_paper_id": ref.split("/")[-1],
            })

        seen_papers.add(w["id"])

    print(f"papers_rows: {len(papers_rows)}")
    print(f"citation_rows: {len(citation_rows)}")
    
    papers_df = pd.DataFrame(papers_rows)
    citations_df = pd.DataFrame(citation_rows)

    return papers_df, citations_df
    

if __name__ == "__main__":
    keywords = ["transformer", "diffusion", "language", "tuning", "retrieval", "neural", "attention", "generative", "training", "instruct"]

    papers_df, citations_df = harvest_openalex_by_keywords(
        keywords=keywords,
        filters={"from_publication_date": "2014-01-01", "topics.subfield.id": "1702"},
        max_records=1000
    )

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    papers_df.to_csv(os.path.join(data_dir, "papers.csv"), index=False)
    citations_df.to_csv(os.path.join(data_dir, "citations.csv"), index=False)