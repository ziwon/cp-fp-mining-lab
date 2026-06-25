import pandas as pd
from PIL import Image

from cv_fp_lab.dataset_builder import (
    _label_path_for,
    build_hard_negative_dataset,
    validate_label_lines,
)


def test_label_path_derivation() -> None:
    p = _label_path_for("/data/D-Fire/train/images/x.jpg")
    assert p.as_posix() == "/data/D-Fire/train/labels/x.txt"


def test_validate_label_lines(tmp_path) -> None:
    f = tmp_path / "a.txt"
    f.write_text(
        "0 0.5 0.5 0.2 0.2\n"      # valid
        "1 0.1 0.1 0.05 0.05\n"     # valid
        "2 0.5 0.5 0.1 0.1\n"       # invalid class (n_classes=2)
        "0 1.5 0.5 0.1 0.1\n"       # out-of-bounds coord
        "0 0.5 0.5 0.1\n"           # wrong arity
    )
    valid, invalid = validate_label_lines(f, n_classes=2)
    assert len(valid) == 2
    assert invalid == 3


def test_build_hard_negatives(tmp_path) -> None:
    # source dataset: one positive image (with GT) + one negative image
    src = tmp_path / "train"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir(parents=True)
    for name in ("pos", "neg"):
        Image.new("RGB", (64, 64)).save(src / "images" / f"{name}.jpg")
    (src / "labels" / "pos.txt").write_text("0 0.5 0.5 0.3 0.3\n9 0.5 0.5 0.1 0.1\n")  # 1 valid + 1 bad class
    (src / "labels" / "neg.txt").write_text("")  # negative image

    events = pd.DataFrame(
        {
            "source_image_path": [
                str(src / "images" / "pos.jpg"),
                str(src / "images" / "neg.jpg"),
                str(src / "images" / "neg.jpg"),  # duplicate source -> one output
                str(src / "images" / "missing.jpg"),
            ],
            "operator_feedback": [
                "false_positive",
                "false_positive",
                "false_positive",
                "false_positive",
            ],
        }
    )
    out = tmp_path / "hardneg"
    stats = build_hard_negative_dataset(events, out, n_classes=2)

    assert stats["n_images"] == 2  # deduped
    assert stats["n_negatives"] == 1  # the empty-label image
    assert stats["n_invalid_labels_dropped"] == 1  # class 9 dropped from pos
    assert stats["n_missing_sources"] == 1
    assert (out / "images" / "pos.jpg").exists()
    # pos label keeps only the valid line; neg label is empty
    assert (out / "labels" / "pos.txt").read_text().strip() == "0 0.5 0.5 0.3 0.3"
    assert (out / "labels" / "neg.txt").read_text().strip() == ""


def test_build_filters_unconfirmed(tmp_path) -> None:
    src = tmp_path / "train"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(src / "images" / "a.jpg")
    (src / "labels" / "a.txt").write_text("")
    events = pd.DataFrame(
        {"source_image_path": [str(src / "images" / "a.jpg")], "operator_feedback": ["real_event"]}
    )
    stats = build_hard_negative_dataset(events, tmp_path / "out", n_classes=2)
    assert stats["n_images"] == 0  # not a confirmed FP


def test_build_disambiguates_colliding_source_names(tmp_path) -> None:
    for split, label in (("a", "0 0.5 0.5 0.3 0.3\n"), ("b", "1 0.5 0.5 0.2 0.2\n")):
        root = tmp_path / split
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir(parents=True)
        Image.new("RGB", (64, 64)).save(root / "images" / "same.jpg")
        (root / "labels" / "same.txt").write_text(label)

    events = pd.DataFrame(
        {
            "source_image_path": [
                str(tmp_path / "a" / "images" / "same.jpg"),
                str(tmp_path / "b" / "images" / "same.jpg"),
            ],
            "operator_feedback": ["false_positive", "false_positive"],
        }
    )

    out = tmp_path / "out"
    stats = build_hard_negative_dataset(events, out, n_classes=2)

    assert stats["n_images"] == 2
    assert len(list((out / "images").glob("same*.jpg"))) == 2
    assert len(list((out / "labels").glob("same*.txt"))) == 2
