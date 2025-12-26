from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import TOP_K_DEFAULT


def _build_memory_index(embeddings: np.ndarray):
    import faiss

    emb = np.array(embeddings, dtype="float32", copy=True)
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def precision_at_k(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    k: int = TOP_K_DEFAULT,
    sample_size: Optional[int] = 500,
) -> Dict[str, float]:
    if sample_size is None or sample_size > len(df):
        sample_size = len(df)
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(df), size=sample_size, replace=False)

    index = _build_memory_index(embeddings)
    precisions, recalls, mrrs, ndcgs = [], [], [], []
    cover_hits = 0
    for idx in sample_indices:
        scores, neighbors = index.search(embeddings[idx : idx + 1], k + 1)
        neighbor_ids = [n for n in neighbors[0].tolist() if n != idx][:k]

        target_cats = set(df.iloc[idx]["category_list"]) if "category_list" in df.columns else set()
        hits = 0
        rels = []
        for n_id in neighbor_ids:
            neigh_cats = set(df.iloc[n_id]["category_list"]) if "category_list" in df.columns else set()
            overlap = target_cats & neigh_cats if target_cats else set()
            rel = 1 if overlap else 0
            rels.append(rel)
            hits += rel
        precisions.append(hits / k if k else 0)
        recalls.append(min(1.0, hits / len(target_cats)) if target_cats else 0)
        try:
            first_rel = rels.index(1)
            mrrs.append(1 / (first_rel + 1))
        except ValueError:
            mrrs.append(0.0)
        dcg = sum(rel / np.log2(rank + 2) for rank, rel in enumerate(rels))
        ideal = sum(1 / np.log2(rank + 2) for rank in range(min(len(target_cats), k))) if target_cats else 0
        ndcgs.append(dcg / ideal if ideal > 0 else 0)
        cover_hits += int(bool(target_cats))

    return {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "mrr": float(np.mean(mrrs)),
        "ndcg": float(np.mean(ndcgs)),
        "category_coverage": float(cover_hits / sample_size),
        "sample_size": int(sample_size),
        "k": int(k),
    }

