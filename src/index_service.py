import numpy as np

try:
    import faiss
except ImportError as exc:
    raise ImportError("faiss-cpu 未安装，请先安装依赖。") from exc

from .config import EMBEDDINGS_PATH, INDEX_PATH, TOP_K_DEFAULT
from .embedding_service import load_embeddings


def _normalize(vecs: np.ndarray) -> np.ndarray:
    faiss.normalize_L2(vecs)
    return vecs


def build_index(embeddings: np.ndarray, index_path=INDEX_PATH) -> "faiss.Index":
    emb = np.array(embeddings, dtype="float32", copy=True)
    _normalize(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(index_path))
    return index


def load_index(index_path=INDEX_PATH) -> "faiss.Index":
    return faiss.read_index(str(index_path))


def search(index: "faiss.Index", query_embeddings: np.ndarray, top_k: int = TOP_K_DEFAULT):
    queries = np.array(query_embeddings, dtype="float32", copy=True)
    _normalize(queries)
    scores, neighbors = index.search(queries, top_k)
    return scores, neighbors


def load_embeddings_and_index():
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    index = load_index(INDEX_PATH)
    return embeddings, index

