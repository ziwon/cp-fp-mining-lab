from __future__ import annotations

import datetime as _dt
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .active_learning import uncertainty_scores


def _new_version() -> str:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"detector-{stamp}-{uuid.uuid4().hex[:6]}"


class FpDetector:
    """False-positive type classifier over event embeddings.

    Predicts the ``fp_type`` of a flagged event from its embedding. This is the
    servable "model" in the active-learning loop: it pre-labels review tasks,
    exposes per-task uncertainty, and is retrained as humans correct labels.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        classes: list[str],
        model_version: str,
        metrics: dict[str, float],
        trained_at: str,
    ) -> None:
        self.pipeline = pipeline
        self.classes = list(classes)
        self.model_version = model_version
        self.metrics = dict(metrics)
        self.trained_at = trained_at

    @classmethod
    def train(
        cls,
        embeddings: np.ndarray,
        labels: list[str] | np.ndarray,
        test_size: float = 0.25,
        seed: int = 42,
    ) -> "FpDetector":
        """Fit a classifier and record held-out metrics.

        Evaluates on a stratified hold-out (when there are enough samples per
        class), then refits on all data so the served model uses every label.
        """
        x = np.asarray(embeddings, dtype=np.float64)
        y = np.asarray(labels)
        if len(x) == 0:
            raise ValueError("Cannot train on empty embeddings.")
        classes = sorted(set(y.tolist()))

        def _make() -> Pipeline:
            clf = (
                DummyClassifier(strategy="most_frequent")
                if len(classes) == 1
                else LogisticRegression(max_iter=1000, C=1.0)
            )
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", clf),
                ]
            )

        metrics: dict[str, float] = {"n_samples": float(len(x)), "n_classes": float(len(classes))}
        _, counts = np.unique(y, return_counts=True)
        n_test = int(np.ceil(test_size * len(x))) if isinstance(test_size, float) else int(test_size)
        can_split = (
            len(classes) > 1
            and counts.min() >= 2
            and n_test >= len(classes)
            and len(x) - n_test >= len(classes)
        )
        if can_split:
            x_tr, x_te, y_tr, y_te = train_test_split(
                x, y, test_size=test_size, random_state=seed, stratify=y
            )
            pred = _make().fit(x_tr, y_tr).predict(x_te)
            metrics["macro_f1"] = float(f1_score(y_te, pred, average="macro", zero_division=0))
            metrics["accuracy"] = float(accuracy_score(y_te, pred))
            metrics["n_eval"] = float(len(y_te))
        else:
            # Not enough data to hold out; report training-fit accuracy as a floor.
            metrics["macro_f1"] = 0.0
            metrics["accuracy"] = 0.0
            metrics["n_eval"] = 0.0

        pipeline = _make().fit(x, y)
        if not can_split:
            pred = pipeline.predict(x)
            metrics["accuracy"] = float(accuracy_score(y, pred))
            metrics["macro_f1"] = float(f1_score(y, pred, average="macro", zero_division=0))

        return cls(
            pipeline=pipeline,
            classes=list(pipeline.named_steps["clf"].classes_),
            model_version=_new_version(),
            metrics=metrics,
            trained_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        )

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(np.asarray(embeddings, dtype=np.float64))

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(np.asarray(embeddings, dtype=np.float64))

    def confidence(self, embeddings: np.ndarray) -> np.ndarray:
        """Top-class probability per sample."""
        return self.predict_proba(embeddings).max(axis=1)

    def uncertainty(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalized entropy over the predicted class distribution ([0, 1])."""
        proba = self.predict_proba(embeddings)
        if proba.shape[1] <= 1:
            return np.zeros(len(proba), dtype=np.float64)
        eps = 1e-12
        ent = -(proba * np.log2(proba + eps)).sum(axis=1)
        return np.clip(ent / np.log2(proba.shape[1]), 0.0, 1.0)

    def binary_uncertainty(self, embeddings: np.ndarray, strategy: str = "entropy") -> np.ndarray:
        """Uncertainty of the top-class confidence via the active-learning scorer."""
        return uncertainty_scores(self.confidence(embeddings), strategy)

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "classes": self.classes,
                "model_version": self.model_version,
                "metrics": self.metrics,
                "trained_at": self.trained_at,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FpDetector":
        import joblib

        blob: dict[str, Any] = joblib.load(Path(path))
        return cls(
            pipeline=blob["pipeline"],
            classes=blob["classes"],
            model_version=blob["model_version"],
            metrics=blob["metrics"],
            trained_at=blob["trained_at"],
        )
