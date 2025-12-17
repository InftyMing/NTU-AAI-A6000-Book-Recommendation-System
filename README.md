# 猜你喜欢 —— 书籍内容推荐系统

基于 SBERT 语义向量、FAISS 余弦检索 + BM25 的混合召回，实现“输入一句话/选一本书，找相似书籍”的本地可运行 Demo。

## 环境准备
```bash
cd /Users/inftyming/Documents/NTU_AAI_Study/sem1-final/NTU-AAI-A6000-Book-Recommendation-System
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据获取
已在 `data/raw/google_books_dataset.csv`（Kaggle 数据集自动下载）。如需重下，运行：
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

## 一键跑全流程
```bash
python -m src.pipeline --step all --sample-size 400 --k 5 --plot docs/metrics.png
# 若 Matplotlib 缓存权限警告，可：
# MPLCONFIGDIR=/tmp/mplcache python -m src.pipeline --step all --sample-size 400 --k 5 --plot docs/metrics.png
```
产物：
- 预处理：`data/processed/books.parquet`
- 语义向量：`data/processed/book_embeddings.npy`
- FAISS 索引：`data/index/books.faiss`
- BM25 索引：`data/index/bm25.pkl`
- 评估图：`docs/metrics.png`

## 单独步骤（可选）
- 预处理：`python -m src.pipeline --step preprocess`
- 向量化：`python -m src.pipeline --step embed`
- BM25：`python -m src.pipeline --step bm25`
- FAISS：`python -m src.pipeline --step index`
- 评估：`python -m src.pipeline --step evaluate --plot docs/metrics.png`

## 本地前端
```bash
streamlit run app.py
```
前端支持语义 / BM25 / 混合检索，混合权重可调；可输入自然语言或从书单选择一本找相似。

## 纯代码检索示例
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

query = "吸血鬼 校园 恋爱"
q = model.encode([query], normalize_embeddings=True)
scores, idxs = search(index, q, top_k=5)
for s, i in zip(scores[0], idxs[0]):
    print(f"{s:.3f}", df.iloc[i]["title"])
PY
```

## 评估指标（k=5, sample=400）
precision@k=0.332, recall@k=0.627, MRR=0.477, nDCG=0.991, category_coverage=0.885（详见 `docs/metrics.png` 与 `docs/report.md`）。

## 目录结构（核心）
- `src/data_utils.py`：清洗与特征构造  
- `src/embedding_service.py`：SBERT 向量化  
- `src/bm25_service.py`：BM25 召回  
- `src/index_service.py`：FAISS 索引与检索  
- `src/evaluation.py`：precision/recall/MRR/nDCG  
- `src/pipeline.py`：全流程 CLI  
- `app.py`：Streamlit 前端  
- `docs/report.md`：项目报告；`docs/metrics.png`：评估图

