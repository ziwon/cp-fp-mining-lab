import pandas as pd
from PIL import Image

from cv_fp_lab.wandb_logging import _flatten_metrics
from cv_fp_lab.wandb_logging import _count_yolo_dataset, _summarize_fp_events


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


def test_summarize_fp_events_counts_classes_and_negative_images(tmp_path) -> None:
    events = tmp_path / "fp_events.csv"
    pd.DataFrame(
        {
            "pred_class": ["fire", "smoke", "fire"],
            "is_negative_image": [True, False, True],
        }
    ).to_csv(events, index=False)

    summary = _summarize_fp_events(events)

    assert summary["n_fp_events"] == 3.0
    assert summary["n_negative_image_fps"] == 2.0
    assert summary["n_pred_class_fire"] == 2.0
    assert summary["n_pred_class_smoke"] == 1.0


def test_count_yolo_dataset_counts_empty_labels(tmp_path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (16, 16)).save(images / "a.jpg")
    Image.new("RGB", (16, 16)).save(images / "b.png")
    (labels / "a.txt").write_text("", encoding="utf-8")
    (labels / "b.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    summary = _count_yolo_dataset(tmp_path)

    assert summary == {
        "n_images": 2.0,
        "n_label_files": 2.0,
        "n_empty_labels": 1.0,
    }
