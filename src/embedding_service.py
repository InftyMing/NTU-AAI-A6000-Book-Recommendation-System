import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from tqdm import tqdm

from .config import BATCH_SIZE, EMBEDDINGS_PATH, MODEL_NAME, PROCESSED_PATH, ensure_dirs
from .data_utils import load_processed


class EmbeddingService:
    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None):
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Iterable[str], batch_size: int = BATCH_SIZE, normalize: bool = True) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=True,
            convert_to_numpy=True,
        )


def build_embeddings(
    processed_path: Path = PROCESSED_PATH,
    output_path: Path = EMBEDDINGS_PATH,
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    ensure_dirs()
    df = load_processed(processed_path)
    service = EmbeddingService(model_name=model_name)
    embeddings = service.encode(df["text"].tolist(), batch_size=batch_size, normalize=True)
    np.save(output_path, embeddings)
    return embeddings


def load_embeddings(path: Path = EMBEDDINGS_PATH) -> np.ndarray:
    if not Path(path).exists():
        raise FileNotFoundError(f"Embeddings not found at {path}. Please run embedding step first.")
    return np.load(path)

