from pathlib import Path

# 项目路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data/raw/google_books_dataset.csv"
PROCESSED_PATH = BASE_DIR / "data/processed/books.parquet"
EMBEDDINGS_PATH = BASE_DIR / "data/processed/book_embeddings.npy"
INDEX_PATH = BASE_DIR / "data/index/books.faiss"
BM25_PATH = BASE_DIR / "data/index/bm25.pkl"

# 模型及参数
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # 768 维
BATCH_SIZE = 64
TOP_K_DEFAULT = 5


def ensure_dirs() -> None:
    """确保输出目录存在。"""
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)

