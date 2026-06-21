from cv_fp_lab.fp_mining import is_false_positive, iou_xyxy, read_yolo_labels


def test_iou_basic() -> None:
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # half-overlap on x: intersection 5x10=50, union 100+100-50=150
    assert abs(iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) - (50 / 150)) < 1e-9


def test_read_yolo_labels(tmp_path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.05 0.05\n")
    rows = read_yolo_labels(p)
    assert rows == [(0, 0.5, 0.5, 0.2, 0.2), (1, 0.1, 0.1, 0.05, 0.05)]
    # missing / empty file -> negative image
    assert read_yolo_labels(tmp_path / "missing.txt") == []


def test_detection_on_negative_image_is_fp() -> None:
    det = {"class_id": 1, "xyxy": [10, 10, 20, 20]}
    assert is_false_positive(det, {}) is True  # no GT at all


def test_matching_detection_is_not_fp() -> None:
    det = {"class_id": 0, "xyxy": [0, 0, 10, 10]}
    gt = {0: [(0, 0, 10, 10)]}
    assert is_false_positive(det, gt, iou_thr=0.4) is False


def test_wrong_class_overlap_is_fp() -> None:
    # Predicts fire (1) where ground truth is smoke (0) at the same place.
    det = {"class_id": 1, "xyxy": [0, 0, 10, 10]}
    gt = {0: [(0, 0, 10, 10)]}
    assert is_false_positive(det, gt, iou_thr=0.4) is True
    # class-agnostic localization check: overlap with any GT -> not an FP
    assert is_false_positive(det, gt, iou_thr=0.4, class_agnostic=True) is False


def test_low_overlap_same_class_is_fp() -> None:
    det = {"class_id": 0, "xyxy": [0, 0, 10, 10]}
    gt = {0: [(8, 8, 18, 18)]}  # tiny overlap, IoU well below 0.4
    assert is_false_positive(det, gt, iou_thr=0.4) is True
