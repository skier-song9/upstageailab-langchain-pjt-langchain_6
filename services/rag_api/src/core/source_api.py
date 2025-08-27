import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", "..", "..", ".."))
print(ROOT_DIR)
sys.path.append(ROOT_DIR)

from db.meta_openalex import search_works_by_keywords
from db.util import reconstruct_abstract

def mock_web_search(paper_title: str) -> dict | None:
    """웹 검색 API를 호출하는 모의 함수."""
    print(f"🌐 웹 검색: '{paper_title}'")
    if "graph rag" in paper_title.lower():
        return {"title": "Graph RAG", "source": "Web", "details": "웹에서 찾은 Graph RAG 정보"}
    return None

def openalex_search(paper_title: str) -> dict | None:
  """
  입력한 논문 제목에 대해 openalex api를 통해 논문 정보를 불러오는 함수
  return fields: 
    id: openalex_id,
    display_name: 논문 제목,
    publication_date: 발행 날짜,
    abstract_inverted_index: 초록(키워드 인덱스),
    authorships: 저자 정보
  """
  filter = {
    "has_abstract":"true",
    "is_paratext":"false",
  }
  select="id,display_name,publication_date,doi,cited_by_count,primary_location,abstract_inverted_index,authorships,referenced_works"
  r = search_works_by_keywords(
    query = paper_title,
    filters = filter,
    select_fields = select,
    max_records = 5,
    per_page = 5,
    rps_sleep = 0,
  )

  if not r:
    return None

  paper = r[0]  
  paper_info = {
    "openalex_id": paper["id"].split("/")[-1],
    "title": paper["display_name"],
    "publication_date": paper["publication_date"],
    "doi": paper["doi"],
    "cited_by_count": paper["cited_by_count"],
    "pdf_url": (paper.get("primary_location") or {}).get("pdf_url"),
    "abstract": reconstruct_abstract(paper["abstract_inverted_index"]),
    "authors": ", ".join([a["author"]["display_name"] for a in paper["authorships"]]),
    "cited_papers": [ref["referenced_works"].split("/")[-1] for ref in paper["referenced_works"]],
  }

  return paper_info

if __name__ == "__main__":
  openalex_search("Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension")
