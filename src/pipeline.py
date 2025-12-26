import argparse
import os
from pathlib import Path
from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

from . import config
from .data_utils import load_processed, preprocess_and_save
from .embedding_service import build_embeddings, load_embeddings
from .evaluation import precision_at_k
from .plot_utils import plot_metrics_bar
from .bm25_service import build_and_save_bm25
from .index_service import build_index


def run_preprocess(raw_path: Path, output_path: Path):
    df = preprocess_and_save(raw_path, output_path)
    print(f"Preprocessing completed, saved to {output_path}, samples: {len(df)}")
    return df


def run_embed(processed_path: Path, output_path: Path):
    embeddings = build_embeddings(processed_path, output_path, model_name=config.MODEL_NAME, batch_size=config.BATCH_SIZE)
    print(f"Embeddings generated, shape={embeddings.shape}, saved to {output_path}")
    return embeddings


def run_bm25(df):
    bm25 = build_and_save_bm25(df, output_path=config.BM25_PATH)
    print(f"BM25 index completed, saved to {config.BM25_PATH}")
    return bm25


def run_index(embeddings_path: Path, index_path: Path):
    embeddings = load_embeddings(embeddings_path)
    index = build_index(embeddings, index_path)
    print(f"Index built, saved to {index_path}, vectors: {index.ntotal}")
    return index


def run_eval(processed_path: Path, embeddings: np.ndarray, k: int, sample_size: Optional[int], plot_path: Optional[Path] = None):
    df = load_processed(processed_path)
    metrics = precision_at_k(df, embeddings, k=k, sample_size=sample_size)
    print("Evaluation metrics:", metrics)
    if plot_path:
        plot_metrics_bar(metrics, plot_path)
        print(f"Evaluation chart saved to {plot_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Book recommendation pipeline")
    parser.add_argument(
        "--step",
        choices=["preprocess", "embed", "index", "bm25", "evaluate", "all"],
        default="all",
        help="Pipeline step",
    )
    parser.add_argument("--k", type=int, default=config.TOP_K_DEFAULT, help="Top-K retrieval")
    parser.add_argument("--sample-size", type=int, default=500, help="Evaluation sample size, None for full dataset")
    parser.add_argument("--plot", type=str, default="docs/metrics.png", help="Evaluation plot output path, empty to skip")
    args = parser.parse_args()

    if args.step in ("preprocess", "all"):
        df = run_preprocess(config.RAW_DATA_PATH, config.PROCESSED_PATH)
    else:
        df = load_processed(config.PROCESSED_PATH)

    if args.step in ("embed", "all"):
        embeddings = run_embed(config.PROCESSED_PATH, config.EMBEDDINGS_PATH)
    else:
        embeddings = load_embeddings(config.EMBEDDINGS_PATH)

    if args.step in ("bm25", "all"):
        run_bm25(df)

    if args.step in ("index", "all"):
        run_index(config.EMBEDDINGS_PATH, config.INDEX_PATH)

    if args.step in ("evaluate", "all"):
        plot_path = Path(args.plot) if args.plot else None
        run_eval(config.PROCESSED_PATH, embeddings, k=args.k, sample_size=args.sample_size, plot_path=plot_path)


if __name__ == "__main__":
    main()

