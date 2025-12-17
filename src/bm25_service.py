import re
from typing import List, Sequence

import joblib
from rank_bm25 import BM25Okapi

from .config import BM25_PATH
from .data_utils import load_processed


def _tokenize(text: str) -> List[str]:
    # 简单分词：仅保留字母与数字，空白切分
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    return [t for t in cleaned.split() if t]


def build_bm25(corpus: Sequence[str]) -> BM25Okapi:
    tokenized = [_tokenize(t) for t in corpus]
    return BM25Okapi(tokenized)


def build_and_save_bm25(df=None, output_path=BM25_PATH):
    if df is None:
        df = load_processed()
    bm25 = build_bm25(df["text"].tolist())
    joblib.dump(bm25, output_path)
    return bm25


def load_bm25(path=BM25_PATH) -> BM25Okapi:
    return joblib.load(path)


def search_bm25(bm25: BM25Okapi, query: str, top_k: int):
    tokenized_q = _tokenize(query)
    scores = bm25.get_scores(tokenized_q)
    # 返回 (score, idx)
    ranking = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return ranking

