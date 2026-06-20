import numpy as np
import pandas as pd

from cv_fp_lab.training import assemble_training_data, persist_review_labels


def _make_dataset(tmp_path, n=12):
    events = pd.DataFrame(
        {
            "event_id": [f"evt_{i}" for i in range(n)],
            "synthetic_fp_type": ["steam" if i % 2 else "fog" for i in range(n)],
        }
    )
    events_csv = tmp_path / "fp_events.csv"
    events.to_csv(events_csv, index=False)
    emb = tmp_path / "emb.npy"
    np.save(emb, np.random.default_rng(0).normal(size=(n, 4)))
    return events_csv, emb


def test_persist_review_labels_merges_and_overwrites(tmp_path) -> None:
    path = tmp_path / "webhook_reviews.csv"
    persist_review_labels(path, {"evt_0": "steam", "evt_1": "fog", "evt_2": None})
    df = pd.read_csv(path)
    assert set(df["event_id"]) == {"evt_0", "evt_1"}  # None dropped

    # Second batch adds one and re-reviews evt_0.
    persist_review_labels(path, {"evt_0": "shadow", "evt_3": "animal"})
    df = pd.read_csv(path)
    got = dict(zip(df["event_id"], df["review_fp_type"]))
    assert got == {"evt_0": "shadow", "evt_1": "fog", "evt_3": "animal"}


def test_assemble_uses_review_labels_over_bootstrap(tmp_path) -> None:
    events_csv, emb = _make_dataset(tmp_path)
    reviews = tmp_path / "webhook_reviews.csv"
    # Correct evt_0 (bootstrap "fog") to "animal".
    persist_review_labels(reviews, {"evt_0": "animal"})

    _, labels_boot, ids = assemble_training_data(events_csv, emb)
    _, labels_rev, _ = assemble_training_data(events_csv, emb, reviewed_csv=reviews)

    i = ids.index("evt_0")
    assert labels_boot[i] == "fog"
    assert labels_rev[i] == "animal"
