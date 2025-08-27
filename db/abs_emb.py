import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT_DIR, "data")

IN_CSV = os.path.join(data_dir, "papers.csv")
OUT_NPY = os.path.join(data_dir, "papers_embeddings.npy")

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 32
MAX_SEQ_LEN = 512           # ★ OOM 방지 핵심
WIN_SIZE = 512              # 슬라이딩 윈도우 토큰 길이
WIN_STRIDE = 448            # 겹침(= WIN_SIZE - overlap)

def tokenize(model, texts, max_len=None):
    tok = model.tokenizer(
        texts,
        padding=True,
        truncation=(max_len is not None),
        max_length=max_len,
        return_tensors="pt"
    )
    return tok

def chunk_by_tokens(model, text, win_size=256, stride=224):
    # 토크나이저로 한 번 토큰화해서 길이 확인
    ids = model.tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False)["input_ids"][0]
    L = ids.size(0)
    if L <= win_size:
        return [text]
    # 여러 chunk로 쪼갠 뒤 개별적으로 다시 디코드하여 encode에 넣음
    chunks = []
    start = 0
    while start < L:
        end = min(start + win_size, L)
        piece_ids = ids[start:end].unsqueeze(0)
        piece_text = model.tokenizer.decode(piece_ids[0], skip_special_tokens=True)
        chunks.append(piece_text)
        if end == L:
            break
        start += stride
    return chunks

def encode_texts(model, texts):
    """
    긴 텍스트는 토큰 윈도우로 쪼개서 개별 임베딩 후 평균.
    짧은 텍스트는 바로 encode. fp16 + inference_mode + autocast 사용.
    """
    # 긴/짧은 나누기
    short_texts = []
    long_groups = []   # (idx, [chunks...])
    for idx, t in enumerate(texts):
        if not t.strip():
            short_texts.append((idx, ""))  # 빈 텍스트
            continue
        ids_len = len(model.tokenizer(t, truncation=False, add_special_tokens=False)["input_ids"])
        if ids_len <= MAX_SEQ_LEN:
            short_texts.append((idx, t))
        else:
            chunks = chunk_by_tokens(model, t, WIN_SIZE, WIN_STRIDE)
            long_groups.append((idx, chunks))

    dim = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.half()  # ★ fp16

    # 결과 버퍼
    results = [None] * len(texts)

    # 1) 짧은 텍스트 일괄 배치 인퍼런스
    short_only = [t for _, t in short_texts]
    idx_only   = [i for i, _ in short_texts]
    # 배치로 나눠서
    for s in range(0, len(short_only), BATCH_SIZE):
        batch = short_only[s:s+BATCH_SIZE]
        if not batch:
            continue
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                embs = model.encode(
                    batch,
                    batch_size=BATCH_SIZE,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False
                ).astype("float32")
        if dim is None and embs.shape[0] > 0:
            dim = embs.shape[1]
        for j, vec in enumerate(embs):
            results[idx_only[s+j]] = vec
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 2) 긴 텍스트는 chunk별로 순차 처리 후 평균
    for idx, chunks in tqdm(long_groups, desc="long texts"):
        chunk_vecs = []
        for c_start in range(0, len(chunks), BATCH_SIZE):
            c_batch = chunks[c_start:c_start+BATCH_SIZE]
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    embs = model.encode(
                        c_batch,
                        batch_size=BATCH_SIZE,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    ).astype("float32")
            chunk_vecs.append(embs)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if chunk_vecs:
            vec = np.vstack(chunk_vecs).mean(axis=0)  # 평균 풀링
            if dim is None:
                dim = vec.shape[0]
            results[idx] = vec
        else:
            # 안전장치: chunk 실패 시 0-벡터
            results[idx] = np.zeros(dim, dtype="float32") if dim else None

    # 빈 텍스트(또는 실패)는 0-벡터 채우기
    if dim is None:
        # 모든 텍스트가 비어있는 극단 케이스 방지
        dim = 1024  # Qwen3-Embedding-0.6B 기본 차원(필요하면 바꿔도 됨)
    for i, v in enumerate(results):
        if v is None:
            results[i] = np.zeros(dim, dtype="float32")
    return np.vstack(results)

def main():
    meta_df = pd.read_csv(IN_CSV)
    abstracts = meta_df["abstract"].fillna("").astype(str).tolist()

    model = SentenceTransformer(MODEL_NAME)
    # ★ encode 내부에서도 잘리지만, 모듈 레벨에서 제한을 강제하는 편이 메모리 안정적
    try:
        model.max_seq_length = MAX_SEQ_LEN
    except Exception:
        pass

    embeddings = encode_texts(model, abstracts)
    np.save(OUT_NPY, embeddings)
    print(f"Saved: {OUT_NPY}, shape={embeddings.shape}, dtype={embeddings.dtype}")

if __name__ == "__main__":
    main()
    
