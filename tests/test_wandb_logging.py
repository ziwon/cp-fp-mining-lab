from cv_fp_lab.wandb_logging import _flatten_metrics


def test_flatten_metrics_includes_per_class_recall() -> None:
    flat = _flatten_metrics(
        "candidate",
        {
            "map50": 0.7,
            "recall_per_class": {"smoke": 0.8, "fire": 0.6},
            "notes": "ignored",
        },
    )

    assert flat == {
        "candidate_map50": 0.7,
        "candidate_recall_per_class_smoke": 0.8,
        "candidate_recall_per_class_fire": 0.6,
    }
