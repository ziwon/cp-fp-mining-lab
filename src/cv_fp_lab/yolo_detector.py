from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# D-Fire class convention (no data.yaml ships with the dataset).
DFIRE_CLASSES = {0: "smoke", 1: "fire"}


def write_dfire_yaml(
    output_path: str | Path,
    train_dir: str | Path,
    val_dir: str | Path,
    classes: dict[int, str] | None = None,
) -> Path:
    """Write an Ultralytics dataset YAML for a YOLO-format fire/smoke set.

    ``train_dir``/``val_dir`` are the split roots containing ``images/`` and
    ``labels/`` (Ultralytics derives the labels path from the images path).
    """
    classes = classes or DFIRE_CLASSES
    output_path = Path(output_path)
    spec = {
        "path": str(Path(train_dir).resolve().parent),
        "train": str((Path(train_dir) / "images").resolve()),
        "val": str((Path(val_dir) / "images").resolve()),
        "names": {int(k): v for k, v in classes.items()},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return output_path


class YoloDetector:
    """Thin adapter over Ultralytics YOLO for the fire/smoke detector.

    Kept deliberately small: train, load, and predict. The mining/registry/gate
    code is framework-agnostic and consumes ``predict`` output, so swapping YOLO
    variants (v8/v11, n/s/m) is a config change, not a code change.
    """

    def __init__(self, model: Any, weights_path: str | None = None) -> None:
        self.model = model
        self.weights_path = weights_path

    @staticmethod
    def _import_yolo():
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise SystemExit(
                "ultralytics is not installed. Run `uv sync --extra detect` "
                "(installs torch + ultralytics)."
            ) from exc
        return YOLO

    @classmethod
    def train(
        cls,
        data_yaml: str | Path,
        base_weights: str = "yolov8n.pt",
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        device: str | int = "cpu",
        project: str | Path = "data/processed/yolo_runs",
        name: str = "dfire",
    ) -> "YoloDetector":
        """Fine-tune from ``base_weights`` on a YOLO dataset and return the model.

        Returns a detector wrapping the best checkpoint. Ultralytics writes the
        run (weights, metrics, plots) under ``project/name``.
        """
        YOLO = cls._import_yolo()
        model = YOLO(base_weights)
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=str(project),
            name=name,
            exist_ok=True,
            verbose=False,
        )
        best = Path(project) / name / "weights" / "best.pt"
        weights = str(best) if best.exists() else None
        if weights:
            model = YOLO(weights)
        return cls(model, weights_path=weights)

    @classmethod
    def load(cls, weights: str | Path) -> "YoloDetector":
        YOLO = cls._import_yolo()
        return cls(YOLO(str(weights)), weights_path=str(weights))

    def predict(
        self,
        image_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
    ) -> list[dict[str, Any]]:
        """Run detection on one image; return normalized-box detections.

        Each detection: ``{class_id, class_name, confidence, xyxy(px), xywhn}``.
        """
        results = self.model.predict(
            source=str(image_path), conf=conf, iou=iou, imgsz=imgsz, verbose=False
        )
        if not results:
            return []
        r = results[0]
        names = r.names
        dets: list[dict[str, Any]] = []
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return dets
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            dets.append(
                {
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id)),
                    "confidence": float(boxes.conf[i].item()),
                    "xyxy": [float(v) for v in boxes.xyxy[i].tolist()],
                    "xywhn": [float(v) for v in boxes.xywhn[i].tolist()],
                }
            )
        return dets
