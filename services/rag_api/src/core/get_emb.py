from functools import lru_cache

import os
import sys
from sentence_transformers import SentenceTransformer

ROOT_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", "..", "..", ".."))
sys.path.append(ROOT_DIR)

from db.abs_emb import encode_texts

@lru_cache(maxsize=1)
def get_emb_model():
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    return model

def get_emb(model, texts: list[str]):
    return encode_texts(model, texts)