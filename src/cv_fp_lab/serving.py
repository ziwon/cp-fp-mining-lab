from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .detector import FpDetector
from .embeddings import simple_image_embedding


def _default_embed(image_path: str) -> np.ndarray:
    return simple_image_embedding(image_path)


def predict_tasks(
    tasks: list[dict[str, Any]],
    detector: FpDetector,
    embed_fn: Callable[[str], np.ndarray] | None = None,
    from_name: str = "fp_type",
    to_name: str = "image",
) -> list[dict[str, Any]]:
    """Map Label Studio tasks to predictions in the ML-backend response format.

    Each task's ``data.image`` is embedded and classified; the result is a Label
    Studio prediction with the chosen ``fp_type``, the model's confidence as the
    score, and an ``uncertainty`` meta field for active-learning sorting.
    """
    embed_fn = embed_fn or _default_embed
    if not tasks:
        return []

    # Prefer the raw local path (openable by the backend) over the UI image ref,
    # which may be a Label Studio local-files URL rather than a file path.
    refs = [t["data"].get("image_local_path") or t["data"]["image"] for t in tasks]
    vectors = np.vstack([embed_fn(r) for r in refs])
    labels = detector.predict(vectors)
    confidences = detector.confidence(vectors)
    uncertainties = detector.uncertainty(vectors)

    predictions: list[dict[str, Any]] = []
    for label, conf, unc in zip(labels, confidences, uncertainties):
        predictions.append(
            {
                "model_version": detector.model_version,
                "score": float(conf),
                "meta": {"uncertainty": float(unc)},
                "result": [
                    {
                        "from_name": from_name,
                        "to_name": to_name,
                        "type": "choices",
                        "value": {"choices": [str(label)]},
                    }
                ],
            }
        )
    return predictions
