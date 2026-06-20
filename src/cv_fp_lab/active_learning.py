from __future__ import annotations

import numpy as np
import pandas as pd


STRATEGIES = ("least_confidence", "margin", "entropy")


def uncertainty_scores(confidences: np.ndarray, strategy: str = "least_confidence") -> np.ndarray:
    """Per-sample uncertainty from a detector confidence treated as P(positive).

    The detector exposes a single confidence ``p`` per event. We treat the
    decision as binary (event vs. not-event), so every strategy peaks at
    ``p = 0.5`` and drops toward confident predictions at ``p = 0`` or ``p = 1``.

    - ``least_confidence``: ``1 - max(p, 1 - p)``
    - ``margin``: ``1 - |p - (1 - p)|``
    - ``entropy``: binary entropy in bits, ``H(p) / log2`` (normalized to [0, 1])
    """
    p = np.asarray(confidences, dtype=np.float64)
    if p.size == 0:
        return np.empty((0,), dtype=np.float64)
    p = np.clip(p, 0.0, 1.0)
    q = 1.0 - p

    if strategy == "least_confidence":
        return 1.0 - np.maximum(p, q)
    if strategy == "margin":
        return 1.0 - np.abs(p - q)
    if strategy == "entropy":
        eps = 1e-12
        ent = -(p * np.log2(p + eps) + q * np.log2(q + eps))
        return np.clip(ent, 0.0, 1.0)
    raise ValueError(f"Unsupported strategy: {strategy!r}. Choose from {STRATEGIES}.")


def rank_with_diversity(
    cluster_ids: np.ndarray,
    scores: np.ndarray,
    budget: int | None = None,
    diversity: bool = True,
) -> np.ndarray:
    """Return row indices ordered by acquisition priority.

    With ``diversity`` enabled the selection round-robins across clusters: each
    round takes the highest-scoring remaining sample from every cluster, so a
    single large cluster cannot monopolize the budget. HDBSCAN noise points
    (``cluster_id == -1``) are treated as singleton clusters, so each competes on
    its own uncertainty rather than being lumped together.

    Without ``diversity`` it is a plain descending sort by score (stable, so ties
    keep input order). ``budget`` truncates the result; ``None`` ranks everything.
    """
    cluster_ids = np.asarray(cluster_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return np.empty((0,), dtype=np.int64)

    limit = n if budget is None else max(0, min(int(budget), n))
    if limit == 0:
        return np.empty((0,), dtype=np.int64)

    if not diversity:
        return np.argsort(-scores, kind="stable")[:limit].astype(np.int64)

    # Group key: real clusters share a key; each noise point is its own group.
    groups: dict[object, list[int]] = {}
    for idx in range(n):
        cid = cluster_ids[idx]
        key = (f"noise_{idx}" if cid == -1 else f"c_{cid}")
        groups.setdefault(key, []).append(idx)

    # Sort each group's members by descending score (stable on ties).
    for key, members in groups.items():
        members.sort(key=lambda i: (-scores[i], i))

    cursors = {key: 0 for key in groups}
    selected: list[int] = []
    while len(selected) < limit:
        # Order groups each round by their current top remaining score.
        round_candidates = [
            (key, members[cursors[key]])
            for key, members in groups.items()
            if cursors[key] < len(members)
        ]
        if not round_candidates:
            break
        round_candidates.sort(key=lambda km: (-scores[km[1]], km[1]))
        for key, idx in round_candidates:
            selected.append(idx)
            cursors[key] += 1
            if len(selected) >= limit:
                break
    return np.array(selected, dtype=np.int64)


def select_for_review(
    df: pd.DataFrame,
    budget: int | None = None,
    strategy: str = "least_confidence",
    diversity: bool = True,
    confidence_col: str = "pred_confidence",
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """Attach uncertainty + acquisition rank and return the review-ordered frame.

    Adds ``uncertainty`` (the score) and ``acquisition_rank`` (0-based review
    order) to every row, then returns the top-``budget`` rows in that order.
    """
    out = df.copy()
    out["uncertainty"] = uncertainty_scores(out[confidence_col].to_numpy(), strategy)
    cluster_ids = (
        out[cluster_col].to_numpy() if cluster_col in out.columns else np.zeros(len(out), dtype=int)
    )
    order = rank_with_diversity(cluster_ids, out["uncertainty"].to_numpy(), budget, diversity)
    ranked = out.iloc[order].copy()
    ranked["acquisition_rank"] = np.arange(len(ranked), dtype=np.int64)
    return ranked
