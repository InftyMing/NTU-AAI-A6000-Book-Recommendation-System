import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from pathlib import Path
from typing import List

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src import config  # noqa: E402
from src.bm25_service import load_bm25, search_bm25  # noqa: E402
from src.data_utils import load_processed  # noqa: E402
from src.embedding_service import EmbeddingService, load_embeddings  # noqa: E402
from src.index_service import load_index, search  # noqa: E402


@st.cache_resource
def load_artifacts():
    df = load_processed(config.PROCESSED_PATH)
    embeddings = load_embeddings(config.EMBEDDINGS_PATH)
    index = load_index(config.INDEX_PATH)
    bm25 = load_bm25(config.BM25_PATH)
    return df, embeddings, index, bm25


@st.cache_resource
def load_model():
    return EmbeddingService(model_name=config.MODEL_NAME).model


def render_results(df, scores, idxs, title: str):
    st.subheader(title)
    rows = []
    for score, idx in zip(scores, idxs):
        idx = int(idx)
        record = df.iloc[idx]
        rows.append(
            {
                "Title": record.get("title", ""),
                "Author": record.get("authors", ""),
                "Categories": ", ".join(record.get("category_list", [])),
                "Similarity": f"{score:.3f}",
                "Description": record.get("description", "")[:200] + "...",
            }
        )
    st.dataframe(rows, use_container_width=True)


def _combine_scores(vec_scores, vec_idxs, bm25_scores, bm25_idxs, alpha: float, top_k: int):
    vec_scores = np.array(vec_scores)
    vec_idxs = np.array(vec_idxs)
    bm25_scores = np.array(bm25_scores)
    bm25_idxs = np.array(bm25_idxs)
    
    if vec_scores.size == 0 and bm25_scores.size == 0:
        return np.array([]), np.array([], dtype=int)
    
    vec_scores = vec_scores.flatten()
    vec_idxs = vec_idxs.flatten().astype(int)
    bm25_scores = bm25_scores.flatten()
    bm25_idxs = bm25_idxs.flatten().astype(int)
    
    if len(vec_scores) != len(vec_idxs) or len(bm25_scores) != len(bm25_idxs):
        return np.array([]), np.array([], dtype=int)
    
    vec_max = vec_scores.max() if len(vec_scores) > 0 and vec_scores.max() > 0 else 1.0
    bm25_max = bm25_scores.max() if len(bm25_scores) > 0 and bm25_scores.max() > 0 else 1.0
    
    merged = {}
    for s, i in zip(vec_scores, vec_idxs):
        idx = int(i)
        if idx >= 0:
            if idx not in merged:
                merged[idx] = 0.0
            merged[idx] += alpha * (s / vec_max if vec_max > 0 else 0)
    
    for s, i in zip(bm25_scores, bm25_idxs):
        idx = int(i)
        if idx >= 0:
            if idx not in merged:
                merged[idx] = 0.0
            merged[idx] += (1 - alpha) * (s / bm25_max if bm25_max > 0 else 0)
    
    if not merged:
        return np.array([]), np.array([], dtype=int)
    
    sorted_items = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]
    scores, idxs = zip(*sorted_items)
    scores_array = np.array(scores)
    max_score = scores_array.max() if len(scores_array) > 0 and scores_array.max() > 0 else 1.0
    normalized_scores = scores_array / max_score if max_score > 0 else scores_array
    return normalized_scores, np.array(idxs, dtype=int)


def search_by_text(query: str, top_k: int, mode: str, alpha: float, model, index, bm25, df):
    if not query.strip():
        st.warning("Please enter a search query")
        return
    if mode in ("semantic", "hybrid"):
        query_emb = model.encode([query], normalize_embeddings=True)
        vec_scores_all, vec_neighbors_all = search(index, np.array(query_emb), top_k=top_k if mode == "semantic" else top_k * 3)
        if vec_scores_all.size > 0 and vec_neighbors_all.size > 0:
            vec_scores = vec_scores_all[0]
            vec_neighbors = vec_neighbors_all[0].astype(int)
        else:
            vec_scores, vec_neighbors = np.array([]), np.array([], dtype=int)
    else:
        vec_scores, vec_neighbors = np.array([]), np.array([], dtype=int)

    if mode in ("bm25", "hybrid"):
        bm25_rank = search_bm25(bm25, query, top_k=top_k if mode == "bm25" else top_k * 3)
        bm25_idxs = np.array([int(i) for i, _ in bm25_rank], dtype=int)
        bm25_scores = np.array([s for _, s in bm25_rank])
    else:
        bm25_idxs, bm25_scores = np.array([], dtype=int), np.array([])

    if mode == "semantic":
        render_results(df, vec_scores, vec_neighbors, "Semantic Retrieval Results")
    elif mode == "bm25":
        render_results(df, bm25_scores, bm25_idxs, "BM25 Retrieval Results")
    else:
        scores, idxs = _combine_scores(
            vec_scores,
            vec_neighbors,
            bm25_scores,
            bm25_idxs,
            alpha=alpha,
            top_k=top_k,
        )
        render_results(df, scores, idxs, "Hybrid Retrieval Results")


def search_by_book(selected_title: str, top_k: int, embeddings, index, df):
    matches = df.index[df["title"] == selected_title].tolist()
    if not matches:
        st.warning("Book not found")
        return
    idx = matches[0]
    scores, neighbors = search(index, embeddings[idx : idx + 1], top_k=top_k + 1)
    neighbor_ids = np.array([int(n) for n in neighbors[0].tolist() if n != idx][:top_k], dtype=int)
    render_results(df, scores[0][1 : top_k + 1], neighbor_ids, f"Books Similar to \"{selected_title}\"")


def main():
    st.title("Book Recommendation System Demo")
    st.caption("Content recommendation based on SBERT 768-dimensional vectors + cosine similarity")

    try:
        df, embeddings, index, bm25 = load_artifacts()
    except FileNotFoundError as exc:
        st.error(f"Missing data or index: {exc}")
        st.info("Please run `python -m src.pipeline --step all` first to generate preprocessed data, embeddings, index and BM25.")
        return

    model = load_model()
    top_k = st.slider("Top K", min_value=3, max_value=20, value=config.TOP_K_DEFAULT)
    mode = st.radio("Retrieval Mode", options=["semantic", "bm25", "hybrid"], format_func=lambda x: {"semantic": "Semantic", "bm25": "BM25", "hybrid": "Hybrid"}[x], index=2)
    alpha = st.slider("Hybrid Weight: Semantic Ratio", min_value=0.1, max_value=0.9, value=0.6, step=0.05, disabled=mode != "hybrid")

    st.markdown("### 1) Natural Language Search")
    query = st.text_input("What kind of book do you want to read?", value="I want to read a book about vampires and campus romance")
    if st.button("Search", type="primary"):
        search_by_text(query, top_k, mode, alpha, model, index, bm25, df)

    st.markdown("### 2) Find Similar Books by Selection")
    selected = st.selectbox("Select a book", options=[""] + df["title"].head(2000).tolist(), index=0)
    if selected:
        search_by_book(selected, top_k, embeddings, index, df)

    st.divider()
    st.write(
        "Tip: To show full book list in dropdown, adjust selectbox sample size in code, but natural language search is recommended."
    )


if __name__ == "__main__":
    main()

