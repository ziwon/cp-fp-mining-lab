import numpy as np

from cv_fp_lab.embeddings import SIMPLE_EMBEDDING_DIM, extract_embeddings


def test_extract_simple_embeddings_handles_empty_input() -> None:
    embeddings = extract_embeddings([], method="simple")

    assert embeddings.shape == (0, SIMPLE_EMBEDDING_DIM)
    assert embeddings.dtype == np.float32
