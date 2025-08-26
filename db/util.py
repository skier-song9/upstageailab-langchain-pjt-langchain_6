import time, random, requests, re
from difflib import SequenceMatcher

OPENALEX = "https://api.openalex.org"
MAILTO   = "wjdqlsrla0309@naver.com"        # 반드시 채우기(폴라이트 풀)
RPS_SLEEP = 2                    # ~8~9 rps
PER_PAGE  = 200                     # works per page (max 200)

def get_json(url, params=None, max_retries=5):
    params = dict(params or {})
    params["mailto"] = MAILTO
    for attempt in range(max_retries):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep((2**attempt) + random.random())
            continue
        r.raise_for_status()
    raise RuntimeError(f"Failed after {max_retries} tries: {url} {params}")

def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'[“”"\'`]', '', s)
    s = re.sub(r'[:.;,!?()\[\]{}]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def title_similarity(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a or not b: return 0.0
    sm = SequenceMatcher(None, a, b).ratio()
    la = abs(len(a) - len(b)) / max(len(a), 1)
    return 0.85*sm - 0.15*la

def reconstruct_abstract(abs_idx: dict) -> str:
    if not abs_idx: return ""
    pos = {}
    for w, idxs in abs_idx.items():
        for i in idxs:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos.keys()))

if __name__ == "__main__":
    params = {
      "search": "attention is all you need",
      "select": "id,display_name,publication_date,doi,cited_by_count,abstract_inverted_index,authorships,primary_location,referenced_works",
    }
    url = f"{OPENALEX}/works"
    r = get_json(url, params)
    print(r.get("results", [])[0])
