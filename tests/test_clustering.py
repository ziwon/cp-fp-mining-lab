import numpy as np

from cv_fp_lab.clustering import cluster_embeddings, reduce_umap


def test_reduce_umap_handles_empty_embeddings() -> None:
    xy = reduce_umap(np.empty((0, 3), dtype=np.float32))

    assert xy.shape == (0, 2)


def test_reduce_umap_handles_single_embedding() -> None:
    xy = reduce_umap(np.ones((1, 3), dtype=np.float32))

    assert xy.shape == (1, 2)
    assert np.allclose(xy, 0)


def test_cluster_embeddings_handles_empty_embeddings() -> None:
    labels = cluster_embeddings(np.empty((0, 3), dtype=np.float32))

    assert labels.shape == (0,)


def test_cluster_embeddings_handles_single_embedding() -> None:
    labels = cluster_embeddings(np.ones((1, 3), dtype=np.float32))

    assert labels.tolist() == [0]
