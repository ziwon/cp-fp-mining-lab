from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def reduce_umap(embeddings: np.ndarray, n_neighbors: int = 10, min_dist: float = 0.05) -> np.ndarray:
    if len(embeddings) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(embeddings) == 1:
        return np.zeros((1, 2), dtype=np.float32)

    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(embeddings)
    except Exception:
        # Fallback to PCA-like SVD if UMAP is unavailable.
        x = StandardScaler().fit_transform(embeddings)
        u, s, _ = np.linalg.svd(x, full_matrices=False)
        return u[:, :2] * s[:2]


def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    fallback_k: int = 6,
) -> np.ndarray:
    if len(embeddings) == 0:
        return np.empty((0,), dtype=np.int64)
    if len(embeddings) == 1:
        return np.zeros((1,), dtype=np.int64)

    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(embeddings)
        if len(set(labels)) > 1:
            return labels
    except Exception:
        pass

    k = min(len(embeddings), fallback_k, max(2, len(embeddings) // 8))
    return KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(embeddings)


def attach_cluster_results(metadata: pd.DataFrame, xy: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    df = metadata.copy()
    df["umap_x"] = xy[:, 0]
    df["umap_y"] = xy[:, 1]
    df["cluster_id"] = labels
    return df
