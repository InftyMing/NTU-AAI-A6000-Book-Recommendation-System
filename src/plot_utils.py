from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def plot_metrics_bar(metrics: Dict[str, float], output_path: Path):
    keys = ["precision_at_k", "recall_at_k", "mrr", "ndcg"]
    vals = [metrics.get(k, 0) for k in keys]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(keys, vals, color="#4e79a7")
    plt.ylim(0, 1)
    plt.title("Retrieval Metrics")
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path

