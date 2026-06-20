from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .detector import FpDetector
from .registry import LocalModelRegistry


def assemble_training_data(
    events_csv: str | Path,
    embeddings_npy: str | Path,
    reviewed_csv: str | Path | None = None,
    bootstrap_label: str = "synthetic_fp_type",
    review_label: str = "review_fp_type",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Assemble (embeddings, labels, event_ids) for detector training.

    Bootstraps labels from the operator-provided ``synthetic_fp_type`` and, when a
    review export is available, overrides them with human ``review_fp_type``
    decisions — the label-refinement that makes retraining improve the model.
    """
    events = pd.read_csv(events_csv)
    embeddings = np.load(embeddings_npy)
    if len(events) != len(embeddings):
        raise ValueError(
            f"events ({len(events)}) and embeddings ({len(embeddings)}) are misaligned."
        )

    labels = events[bootstrap_label].astype(object).copy()
    if reviewed_csv is not None and Path(reviewed_csv).exists():
        reviewed = pd.read_csv(reviewed_csv)
        if review_label in reviewed.columns:
            corrections = (
                reviewed.dropna(subset=[review_label])
                .set_index("event_id")[review_label]
                .to_dict()
            )
            corrections = {k: v for k, v in corrections.items() if v not in (None, "", "unknown")}
            mask = events["event_id"].map(lambda e: e in corrections)
            labels.loc[mask] = events.loc[mask, "event_id"].map(corrections)

    keep = labels.notna().to_numpy()
    return embeddings[keep], labels[keep].to_numpy(), events.loc[keep, "event_id"].tolist()


def persist_review_labels(
    path: str | Path,
    reviews: dict[str, str | None],
    review_label: str = "review_fp_type",
) -> pd.DataFrame:
    """Merge reviewed labels into a CSV the trainer reads, keyed by event_id.

    Accumulates across batches (existing rows are kept) and lets a re-reviewed
    event overwrite its earlier label. Rows without a label are dropped.
    """
    path = Path(path)
    rows = [{"event_id": k, review_label: v} for k, v in reviews.items() if v]
    new = pd.DataFrame(rows, columns=["event_id", review_label])
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
    merged = new.drop_duplicates(subset=["event_id"], keep="last").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return merged


def retrain_and_register(
    registry: LocalModelRegistry,
    events_csv: str | Path,
    embeddings_npy: str | Path,
    reviewed_csv: str | Path | None = None,
    metric_key: str = "macro_f1",
    min_delta: float = 0.0,
) -> dict:
    """Train a fresh detector from current labels and gate-promote it.

    Returns the promotion decision dict from the registry, augmented with the new
    model version and training-set size.
    """
    embeddings, labels, event_ids = assemble_training_data(
        events_csv, embeddings_npy, reviewed_csv
    )
    detector = FpDetector.train(embeddings, labels)
    result = registry.maybe_promote(detector, metric_key=metric_key, min_delta=min_delta)
    result["n_train"] = len(event_ids)
    result["metrics"] = detector.metrics
    return result
