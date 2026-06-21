from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from PIL import Image

from .utils import ensure_dir


class _Detector(Protocol):
    def predict(self, image_path: str | Path, **kwargs: Any) -> list[dict[str, Any]]: ...


def read_yolo_labels(label_path: str | Path) -> list[tuple[int, float, float, float, float]]:
    """Read a YOLO label file → list of (class_id, cx, cy, w, h) (normalized).

    A missing or empty file means a negative image (no ground-truth objects).
    """
    path = Path(label_path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            c, cx, cy, w, h = parts[:5]
            rows.append((int(float(c)), float(cx), float(cy), float(w), float(h)))
    return rows


def _xywhn_to_xyxy(box: tuple[float, float, float, float], w: int, h: int) -> tuple[float, ...]:
    cx, cy, bw, bh = box
    return (
        (cx - bw / 2) * w,
        (cy - bh / 2) * h,
        (cx + bw / 2) * w,
        (cy + bh / 2) * h,
    )


def iou_xyxy(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection-over-union of two pixel boxes (x0, y0, x1, y1)."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def is_false_positive(
    det: dict[str, Any],
    gt_xyxy_by_class: dict[int, list[tuple[float, ...]]],
    iou_thr: float = 0.4,
    class_agnostic: bool = False,
) -> bool:
    """A detection is a false positive if it matches no ground-truth box.

    By default a match requires the same class and IoU >= ``iou_thr`` (a fire
    box predicted where there is only smoke still counts as a confusion-type FP).
    ``class_agnostic`` matches against any-class GT (localization-only check).
    """
    det_box = tuple(det["xyxy"])
    if class_agnostic:
        candidates = [b for boxes in gt_xyxy_by_class.values() for b in boxes]
    else:
        candidates = gt_xyxy_by_class.get(det["class_id"], [])
    return all(iou_xyxy(det_box, gt) < iou_thr for gt in candidates)


def collect_false_positives(
    detector: _Detector,
    images_dir: str | Path,
    labels_dir: str | Path,
    crops_dir: str | Path,
    model_version: str,
    conf: float = 0.25,
    iou_thr: float = 0.4,
    class_names: dict[int, str] | None = None,
    limit: int | None = None,
    min_crop_px: int = 8,
) -> pd.DataFrame:
    """Run the detector over a YOLO split and harvest false-positive crops.

    For every image, detections that match no ground-truth box become mined FP
    events: the crop is saved and a row is emitted in the ``fp_events.csv`` schema
    the rest of the pipeline (embeddings → cluster → review) already consumes, so
    real mined FPs replace the synthetic generator with no downstream changes.
    """
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    crops_dir = ensure_dir(crops_dir)
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if limit is not None:
        image_paths = image_paths[:limit]

    rows: list[dict[str, Any]] = []
    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        with Image.open(img_path) as im:
            w, h = im.size
            gt = read_yolo_labels(label_path)
            gt_by_class: dict[int, list[tuple[float, ...]]] = {}
            for c, cx, cy, bw, bh in gt:
                gt_by_class.setdefault(c, []).append(_xywhn_to_xyxy((cx, cy, bw, bh), w, h))

            dets = detector.predict(img_path, conf=conf)
            fp_idx = 0
            for det in dets:
                if not is_false_positive(det, gt_by_class, iou_thr):
                    continue
                x0, y0, x1, y1 = (int(round(v)) for v in det["xyxy"])
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                if x1 - x0 < min_crop_px or y1 - y0 < min_crop_px:
                    continue
                event_id = f"fp_{img_path.stem}_{fp_idx:02d}"
                crop_path = crops_dir / f"{event_id}.png"
                im.crop((x0, y0, x1, y1)).convert("RGB").save(crop_path)
                # camera/site stand-ins derived from the D-Fire source prefix.
                prefix = "".join(ch for ch in img_path.stem if not ch.isdigit())[:12] or "src"
                rows.append(
                    {
                        "event_id": event_id,
                        "image_path": str(crop_path),
                        "source_image_path": str(img_path),
                        "camera_id": prefix,
                        "site_id": "dfire",
                        "timestamp": "",
                        "pred_class": det["class_name"],
                        "pred_confidence": round(det["confidence"], 4),
                        "bbox_x0": x0,
                        "bbox_y0": y0,
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "operator_feedback": "false_positive",
                        # Detector's (wrong) guess bootstraps the label; the true
                        # type is assigned during human review (review_fp_type).
                        "synthetic_fp_type": det["class_name"],
                        "is_negative_image": len(gt) == 0,
                        "model_version": model_version,
                    }
                )
                fp_idx += 1
    return pd.DataFrame(rows)
