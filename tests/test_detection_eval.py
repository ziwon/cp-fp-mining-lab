from PIL import Image

from cv_fp_lab.detection_eval import false_positive_rate


class _FakeDetector:
    """Returns preset detections keyed by image stem."""

    def __init__(self, by_stem):
        self.by_stem = by_stem

    def predict(self, image_path, **kwargs):
        from pathlib import Path

        return self.by_stem.get(Path(image_path).stem, [])


def _make_image(path, size=(100, 100)):
    Image.new("RGB", size).save(path)


def test_false_positive_rate_on_negatives(tmp_path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()

    # neg1: object-free, detector fires once -> FP
    _make_image(images / "neg1.jpg")
    # neg2: object-free, detector silent -> clean
    _make_image(images / "neg2.jpg")
    # pos1: has a GT box; detector hits it (not an FP)
    _make_image(images / "pos1.jpg")
    (labels / "pos1.txt").write_text("0 0.5 0.5 0.4 0.4\n")  # center box

    det = _FakeDetector(
        {
            "neg1": [{"class_id": 0, "xyxy": [10, 10, 30, 30]}],
            "pos1": [{"class_id": 0, "xyxy": [30, 30, 70, 70]}],  # matches GT
        }
    )
    m = false_positive_rate(det, images, labels, conf=0.25, iou_thr=0.4)

    assert m["n_images"] == 3.0
    assert m["n_negatives"] == 2.0
    assert m["neg_fp_rate"] == 0.5  # 1 of 2 negatives fired
    assert m["mean_fp_per_neg"] == 0.5  # 1 FP box / 2 negatives
    assert m["mean_fp_per_image"] == 1 / 3  # only neg1's box is an FP
