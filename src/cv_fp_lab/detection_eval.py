from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .fp_mining import _xywhn_to_xyxy, is_false_positive, read_yolo_labels


class _Detector(Protocol):
    def predict(self, image_path: str | Path, **kwargs: Any) -> list[dict[str, Any]]: ...


def detection_metrics(
    weights: str | Path,
    data_yaml: str | Path,
    split: str = "val",
    conf: float = 0.25,
    iou: float = 0.5,
    device: str | int = "cpu",
) -> dict[str, Any]:
    """Standard detection metrics on a frozen eval set via Ultralytics val.

    Returns mAP@50, mAP@50-95, mean precision/recall, and per-class recall — the
    quality side of the gate (did retraining keep detecting real events?).
    """
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise SystemExit("ultralytics is not installed. Run `uv sync --extra detect`.") from exc

    res = YOLO(str(weights)).val(
        data=str(data_yaml), split=split, conf=conf, iou=iou, device=device, verbose=False
    )
    names = res.names if isinstance(res.names, dict) else dict(enumerate(res.names))
    recall_per_class = {
        names.get(int(c), str(int(c))): float(res.box.r[i])
        for i, c in enumerate(res.box.ap_class_index)
    }
    return {
        "map50": float(res.box.map50),
        "map5095": float(res.box.map),
        "precision": float(res.box.mp),
        "recall": float(res.box.mr),
        "recall_per_class": recall_per_class,
    }


def false_positive_rate(
    detector: _Detector,
    images_dir: str | Path,
    labels_dir: str | Path,
    conf: float = 0.25,
    iou_thr: float = 0.4,
    limit: int | None = None,
) -> dict[str, float]:
    """False-positive load over an eval split — the cost side of the gate.

    Computes, over scanned images: the FP-rate on negative images (fraction of
    object-free images where the detector fires), the mean number of FP boxes per
    negative image, and mean FP boxes per image overall. Lower is better; this is
    what FP mining is supposed to drive down without hurting recall.
    """
    from PIL import Image

    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if limit is not None:
        paths = paths[:limit]

    n_neg = 0
    neg_with_fp = 0
    fp_on_neg = 0
    fp_total = 0
    for img_path in paths:
        gt = read_yolo_labels(labels_dir / f"{img_path.stem}.txt")
        is_neg = len(gt) == 0
        with Image.open(img_path) as im:
            w, h = im.size
        gt_by_class: dict[int, list[tuple[float, ...]]] = {}
        for c, cx, cy, bw, bh in gt:
            gt_by_class.setdefault(c, []).append(_xywhn_to_xyxy((cx, cy, bw, bh), w, h))
        fps = sum(1 for d in detector.predict(img_path, conf=conf) if is_false_positive(d, gt_by_class, iou_thr))
        fp_total += fps
        if is_neg:
            n_neg += 1
            fp_on_neg += fps
            if fps > 0:
                neg_with_fp += 1

    n = len(paths)
    return {
        "n_images": float(n),
        "n_negatives": float(n_neg),
        "neg_fp_rate": (neg_with_fp / n_neg) if n_neg else 0.0,
        "mean_fp_per_neg": (fp_on_neg / n_neg) if n_neg else 0.0,
        "mean_fp_per_image": (fp_total / n) if n else 0.0,
    }
