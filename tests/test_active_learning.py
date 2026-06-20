import numpy as np
import pandas as pd

from cv_fp_lab.active_learning import (
    rank_with_diversity,
    select_for_review,
    uncertainty_scores,
)


def test_uncertainty_scores_handle_empty() -> None:
    for strategy in ("least_confidence", "margin", "entropy"):
        assert uncertainty_scores(np.empty((0,)), strategy).shape == (0,)


def test_uncertainty_peaks_at_decision_boundary() -> None:
    # p = 0.5 is maximally uncertain; p = 0.95 is confident.
    conf = np.array([0.5, 0.95])
    for strategy in ("least_confidence", "margin", "entropy"):
        scores = uncertainty_scores(conf, strategy)
        assert scores[0] > scores[1]


def test_entropy_is_normalized() -> None:
    scores = uncertainty_scores(np.array([0.5, 0.0, 1.0]), "entropy")
    assert np.isclose(scores[0], 1.0)
    assert scores[1] < 1e-6 and scores[2] < 1e-6


def test_least_confidence_ranks_by_distance_from_one() -> None:
    conf = np.array([0.6, 0.9, 0.55])
    order = rank_with_diversity(np.zeros(3, dtype=int), uncertainty_scores(conf), diversity=False)
    # Lowest confidence (0.55) is most uncertain, then 0.6, then 0.9.
    assert order.tolist() == [2, 0, 1]


def test_diversity_round_robins_across_clusters() -> None:
    # Cluster 0 holds the three highest scores; a plain sort would take all three.
    cluster_ids = np.array([0, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.1])

    plain = rank_with_diversity(cluster_ids, scores, budget=2, diversity=False)
    assert plain.tolist() == [0, 1]  # both from cluster 0

    diverse = rank_with_diversity(cluster_ids, scores, budget=2, diversity=True)
    assert sorted(cluster_ids[diverse]) == [0, 1]  # one from each cluster


def test_noise_points_compete_individually() -> None:
    # Two noise points (-1) must not be collapsed into one group.
    cluster_ids = np.array([-1, -1, 0])
    scores = np.array([0.95, 0.90, 0.10])
    order = rank_with_diversity(cluster_ids, scores, budget=2, diversity=True)
    # Both high-scoring noise points should be selected ahead of the weak cluster.
    assert sorted(order.tolist()) == [0, 1]


def test_budget_truncates_and_none_keeps_all() -> None:
    cluster_ids = np.zeros(5, dtype=int)
    scores = np.arange(5, dtype=float)
    assert len(rank_with_diversity(cluster_ids, scores, budget=3)) == 3
    assert len(rank_with_diversity(cluster_ids, scores, budget=None)) == 5
    assert len(rank_with_diversity(cluster_ids, scores, budget=99)) == 5


def test_select_for_review_attaches_columns_and_orders() -> None:
    df = pd.DataFrame(
        {
            "event_id": ["a", "b", "c"],
            "pred_confidence": [0.95, 0.55, 0.80],
            "cluster_id": [0, 0, 0],
        }
    )
    ranked = select_for_review(df, budget=2, strategy="least_confidence", diversity=False)
    assert list(ranked.columns).count("uncertainty") == 1
    assert ranked["acquisition_rank"].tolist() == [0, 1]
    # Least confident sample (b, 0.55) ranks first.
    assert ranked.iloc[0]["event_id"] == "b"
    assert len(ranked) == 2
