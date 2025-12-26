import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import PROCESSED_PATH, RAW_DATA_PATH, ensure_dirs


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def _split_categories(value: Optional[str]) -> List[str]:
    if not isinstance(value, str):
        return []
    tokens = re.split(r"[|/,;]", value)
    cleaned = [t.strip().lower() for t in tokens if t and t.strip()]
    return cleaned


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = work.drop_duplicates(subset=["book_id"])
    for col in ["title", "subtitle", "authors", "description", "categories", "search_category"]:
        if col in work.columns:
            work[col] = work[col].fillna("").astype(str).str.strip()
    work["text"] = work[["title", "subtitle", "description"]].agg(" ".join, axis=1).str.strip()
    work = work[work["text"].str.len() > 50]

    work["category_list"] = work.get("categories", "").apply(_split_categories)
    work["primary_category"] = work["category_list"].apply(lambda x: x[0] if x else "unknown")

    columns = [
        "book_id",
        "title",
        "subtitle",
        "authors",
        "description",
        "categories",
        "category_list",
        "primary_category",
        "text",
        "average_rating",
        "ratings_count",
        "search_category",
    ]
    existing = [c for c in columns if c in work.columns]
    work = work[existing].reset_index(drop=True)
    return work


def preprocess_and_save(raw_path: Path = RAW_DATA_PATH, output_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    ensure_dirs()
    df = load_raw(raw_path)
    processed = preprocess(df)
    processed.to_parquet(output_path, index=False)
    return processed


def load_processed(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found at {path}. Please run preprocessing first.")
    return pd.read_parquet(path)

