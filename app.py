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
        record = df.iloc[idx]
        rows.append(
            {
                "标题": record.get("title", ""),
                "作者": record.get("authors", ""),
                "类别": ", ".join(record.get("category_list", [])),
                "相似度": f"{score:.3f}",
                "简介": record.get("description", "")[:200] + "...",
            }
        )
    st.dataframe(rows, use_container_width=True)


def _combine_scores(vec_scores, vec_idxs, bm25_scores, bm25_idxs, alpha: float, top_k: int):
    # 归一化后线性融合：alpha * 语义 + (1 - alpha) * BM25
    vec_max = vec_scores.max() if len(vec_scores) else 1
    bm25_max = bm25_scores.max() if len(bm25_scores) else 1
    merged = {}
    for s, i in zip(vec_scores, vec_idxs):
        merged[i] = merged.get(i, 0) + alpha * (s / vec_max)
    for s, i in zip(bm25_scores, bm25_idxs):
        merged[i] = merged.get(i, 0) + (1 - alpha) * (s / bm25_max if bm25_max else 0)
    sorted_items = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]
    scores, idxs = zip(*sorted_items)
    return np.array(scores), np.array(idxs)


def search_by_text(query: str, top_k: int, mode: str, alpha: float, model, index, bm25, df):
    if not query.strip():
        st.warning("请输入搜索语句")
        return
    if mode in ("semantic", "hybrid"):
        query_emb = model.encode([query], normalize_embeddings=True)
        vec_scores, vec_neighbors = search(index, np.array(query_emb), top_k=top_k if mode == "semantic" else top_k * 3)
    else:
        vec_scores, vec_neighbors = np.array([[]]), np.array([[]])

    if mode in ("bm25", "hybrid"):
        bm25_rank = search_bm25(bm25, query, top_k=top_k if mode == "bm25" else top_k * 3)
        bm25_idxs = np.array([i for i, _ in bm25_rank])
        bm25_scores = np.array([s for _, s in bm25_rank])
    else:
        bm25_idxs, bm25_scores = np.array([]), np.array([])

    if mode == "semantic":
        render_results(df, vec_scores[0], vec_neighbors[0], "语义检索结果")
    elif mode == "bm25":
        render_results(df, bm25_scores, bm25_idxs, "BM25 检索结果")
    else:
        scores, idxs = _combine_scores(
            vec_scores[0] if vec_scores.size else np.array([]),
            vec_neighbors[0] if vec_neighbors.size else np.array([]),
            bm25_scores,
            bm25_idxs,
            alpha=alpha,
            top_k=top_k,
        )
        render_results(df, scores, idxs, "混合检索结果")


def search_by_book(selected_title: str, top_k: int, embeddings, index, df):
    matches = df.index[df["title"] == selected_title].tolist()
    if not matches:
        st.warning("未找到该书名")
        return
    idx = matches[0]
    scores, neighbors = search(index, embeddings[idx : idx + 1], top_k=top_k + 1)
    neighbor_ids = [n for n in neighbors[0].tolist() if n != idx][:top_k]
    render_results(df, scores[0][1 : top_k + 1], neighbor_ids, f"与《{selected_title}》相似的书籍")


def main():
    st.title("猜你喜欢：书籍语义推荐 Demo")
    st.caption("基于 SBERT 768 维向量 + 余弦相似度的内容推荐")

    try:
        df, embeddings, index, bm25 = load_artifacts()
    except FileNotFoundError as exc:
        st.error(f"缺少数据或索引：{exc}")
        st.info("请先运行 `python -m src.pipeline --step all` 生成预处理、向量、索引与 BM25。")
        return

    model = load_model()
    top_k = st.slider("Top K", min_value=3, max_value=20, value=config.TOP_K_DEFAULT)
    mode = st.radio("检索方式", options=["semantic", "bm25", "hybrid"], format_func=lambda x: {"semantic": "语义", "bm25": "BM25", "hybrid": "混合"}[x], index=2)
    alpha = st.slider("混合权重：语义占比", min_value=0.1, max_value=0.9, value=0.6, step=0.05, disabled=mode != "hybrid")

    st.markdown("### 1) 自然语言搜索")
    query = st.text_input("想看什么类型的书？", value="想看一本关于吸血鬼和校园恋爱的书")
    if st.button("搜索", type="primary"):
        search_by_text(query, top_k, mode, alpha, model, index, bm25, df)

    st.markdown("### 2) 选择已有书籍找相似")
    selected = st.selectbox("选择一本书", options=[""] + df["title"].head(2000).tolist(), index=0)
    if selected:
        search_by_book(selected, top_k, embeddings, index, df)

    st.divider()
    st.write(
        "提示：如需全量书单下拉，请在代码中调整 selectbox 取样数量，但更推荐通过自然语言搜索。"
    )


if __name__ == "__main__":
    main()

