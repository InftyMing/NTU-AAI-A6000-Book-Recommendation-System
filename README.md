# Book Recommendation System - "Guess What You Like"

A locally runnable demo implementing "input a sentence/select a book, find similar books" based on SBERT semantic vectors, FAISS cosine retrieval + BM25 hybrid retrieval.

## Environment Setup
```bash
cd /Users/inftyming/Documents/NTU_AAI_Study/sem1-final/NTU-AAI-A6000-Book-Recommendation-System
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Acquisition
Already available at `data/raw/google_books_dataset.csv` (automatically downloaded from Kaggle dataset). To re-download, run:
```bash
python - <<'PY'
import kagglehub, shutil
from pathlib import Path
cache = kagglehub.dataset_download("mihikaajayjadhav/books-dataset-15k-books-across-100-categories")
dst = Path("data/raw"); dst.mkdir(parents=True, exist_ok=True)
shutil.copy2(Path(cache) / "google_books_dataset.csv", dst / "google_books_dataset.csv")
print("done:", dst)
PY
```

## Run Complete Pipeline
```bash
python -m src.pipeline --step all --sample-size 400 --k 5 --plot docs/metrics.png
# If Matplotlib cache permission warning:
# MPLCONFIGDIR=/tmp/mplcache python -m src.pipeline --step all --sample-size 400 --k 5 --plot docs/metrics.png
```
Outputs:
- Preprocessed data: `data/processed/books.parquet`
- Semantic vectors: `data/processed/book_embeddings.npy`
- FAISS index: `data/index/books.faiss`
- BM25 index: `data/index/bm25.pkl`
- Evaluation chart: `docs/metrics.png`

## Individual Steps (Optional)
- Preprocessing: `python -m src.pipeline --step preprocess`
- Vectorization: `python -m src.pipeline --step embed`
- BM25: `python -m src.pipeline --step bm25`
- FAISS: `python -m src.pipeline --step index`
- Evaluation: `python -m src.pipeline --step evaluate --plot docs/metrics.png`

## Local Frontend
```bash
streamlit run app.py
```
The frontend supports semantic / BM25 / hybrid retrieval with adjustable hybrid weights; can input natural language or select a book from the list to find similar books.

## Pure Code Retrieval Example
```bash
python - <<'PY'
from src import config
from src.data_utils import load_processed
from src.embedding_service import EmbeddingService, load_embeddings
from src.index_service import load_index, search

df = load_processed(config.PROCESSED_PATH)
emb = load_embeddings(config.EMBEDDINGS_PATH)
index = load_index(config.INDEX_PATH)
model = EmbeddingService(model_name=config.MODEL_NAME).model

query = "vampires campus romance"
q = model.encode([query], normalize_embeddings=True)
scores, idxs = search(index, q, top_k=5)
for s, i in zip(scores[0], idxs[0]):
    print(f"{s:.3f}", df.iloc[i]["title"])
PY
```

## Evaluation Metrics (k=5, sample=400)
precision@k=0.332, recall@k=0.627, MRR=0.477, nDCG=0.991, category_coverage=0.885 (see `docs/metrics.png` and `docs/book_recommendation_system_report.md` for details).

## Directory Structure (Core)
- `src/data_utils.py`: Cleaning and feature construction  
- `src/embedding_service.py`: SBERT vectorization  
- `src/bm25_service.py`: BM25 retrieval  
- `src/index_service.py`: FAISS indexing and retrieval  
- `src/evaluation.py`: precision/recall/MRR/nDCG  
- `src/pipeline.py`: Complete pipeline CLI  
- `app.py`: Streamlit frontend  
- `docs/book_recommendation_system_report.md`: Project report; `docs/metrics.png`: Evaluation chart
