from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


def log_table_and_artifact(project: str, artifact_name: str, processed_dir: str | Path) -> None:
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "wandb is not installed. Run `uv sync` first, "
            "or skip scripts/05_log_to_wandb.py for a local-only demo."
        ) from exc

    processed_dir = Path(processed_dir)
    run = wandb.init(project=project, job_type="fp-curation")

    cluster_csv = processed_dir / "fp_clusters.csv"
    reviewed_csv = processed_dir / "reviewed_fp_samples.csv"
    table_csv = reviewed_csv if reviewed_csv.exists() else cluster_csv

    df = pd.read_csv(table_csv)
    table = wandb.Table(dataframe=df)
    run.log({"false_positive_samples": table})

    artifact = wandb.Artifact(artifact_name, type="dataset")
    for p in processed_dir.glob("*"):
        if p.is_file():
            artifact.add_file(str(p))
    run.log_artifact(artifact)
    run.finish()


def log_retrain_run(
    project: str,
    result: dict,
    registry_dir: str | Path,
    reviews: dict[str, str | None] | None = None,
) -> str | None:
    """Log a webhook-triggered retrain as a W&B run with metrics + model artifact.

    When ``reviews`` (the {event_id: label} batch that triggered the retrain) is
    given, also logs it as a W&B Table so the run shows exactly which human labels
    drove this model. Returns the run URL, or None if wandb is unavailable. Honors
    WANDB_MODE, so it is a no-op-to-disk offline and syncs to the server online.
    """
    try:
        import wandb
    except ModuleNotFoundError:
        return None

    run = wandb.init(project=project, job_type="retrain", reinit=True)
    metrics = dict(result.get("metrics", {}))
    metrics["promoted"] = int(bool(result.get("promoted")))
    if result.get("candidate_metric") is not None:
        metrics[result.get("metric_key", "macro_f1")] = result["candidate_metric"]
    run.summary.update(metrics)
    run.log(metrics)

    if reviews:
        table = wandb.Table(columns=["event_id", "review_fp_type"])
        for event_id, label in sorted(reviews.items()):
            table.add_data(event_id, label)
        run.log({"review_batch": table})

    version = result.get("version")
    model_path = Path(registry_dir) / "models" / str(version) / "model.joblib"
    if version and model_path.exists():
        artifact = wandb.Artifact(str(version), type="model")
        artifact.add_file(str(model_path))
        aliases = ["latest", "production"] if result.get("promoted") else ["latest", "candidate"]
        run.log_artifact(artifact, aliases=aliases)

    url = run.get_url()
    run.finish()
    return url


def _flatten_metrics(prefix: str, metrics: dict[str, Any] | None) -> dict[str, float]:
    flat: dict[str, float] = {}
    if not metrics:
        return flat
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, int | float):
                    flat[f"{prefix}_{key}_{sub_key}"] = float(sub_value)
        elif isinstance(value, int | float):
            flat[f"{prefix}_{key}"] = float(value)
    return flat


def _summarize_fp_events(events_csv: str | Path) -> dict[str, float]:
    path = Path(events_csv)
    if not path.exists():
        return {"n_fp_events": 0.0, "n_negative_image_fps": 0.0}
    df = pd.read_csv(path)
    summary = {"n_fp_events": float(len(df))}
    if "is_negative_image" in df.columns:
        summary["n_negative_image_fps"] = float(df["is_negative_image"].fillna(False).sum())
    if "pred_class" in df.columns:
        for cls, count in df["pred_class"].fillna("unknown").value_counts().items():
            summary[f"n_pred_class_{cls}"] = float(count)
    return summary


def _count_yolo_dataset(dataset_dir: str | Path) -> dict[str, float]:
    root = Path(dataset_dir)
    image_dir = root / "images"
    label_dir = root / "labels"
    images = []
    if image_dir.exists():
        images = [
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    labels = list(label_dir.glob("*.txt")) if label_dir.exists() else []
    empty_labels = 0
    for label in labels:
        if not label.read_text(encoding="utf-8").strip():
            empty_labels += 1
    return {
        "n_images": float(len(images)),
        "n_label_files": float(len(labels)),
        "n_empty_labels": float(empty_labels),
    }


def log_mining_run(
    project: str,
    events_csv: str | Path,
    crops_dir: str | Path,
    weights: str | Path,
    source_images: str | Path,
    source_labels: str | Path,
    conf: float,
    iou_thr: float,
    limit: int | None = None,
) -> str | None:
    """Log mined real false positives and crop artifacts for the YOLO track."""
    try:
        import wandb
    except ModuleNotFoundError:
        return None

    os.environ.setdefault("WANDB_MODE", "offline")
    run = wandb.init(project=project, job_type="mine-fp", reinit=True)
    metrics = _summarize_fp_events(events_csv)
    run.summary.update(metrics)
    run.summary.update(
        {
            "weights": str(weights),
            "source_images": str(source_images),
            "source_labels": str(source_labels),
            "conf": conf,
            "iou_thr": iou_thr,
            "limit": limit,
        }
    )
    run.log(metrics)

    events_path = Path(events_csv)
    if events_path.exists():
        df = pd.read_csv(events_path)
        run.log({"fp_events": wandb.Table(dataframe=df.head(2000))})
        artifact = wandb.Artifact("mined-fp-events", type="dataset")
        artifact.add_file(str(events_path))
        crops_path = Path(crops_dir)
        if crops_path.exists():
            artifact.add_dir(str(crops_path), name="crops")
        run.log_artifact(artifact, aliases=["latest"])

    url = run.get_url()
    run.finish()
    return url


def log_hard_negative_dataset_run(
    project: str,
    events_csv: str | Path,
    hard_negative_dir: str | Path,
    stats: dict[str, Any],
) -> str | None:
    """Log a reviewed/confirmed FP to YOLO hard-negative dataset build."""
    try:
        import wandb
    except ModuleNotFoundError:
        return None

    os.environ.setdefault("WANDB_MODE", "offline")
    run = wandb.init(project=project, job_type="build-hardneg", reinit=True)
    metrics = {k: float(v) for k, v in stats.items() if isinstance(v, int | float)}
    metrics.update(_count_yolo_dataset(hard_negative_dir))
    run.summary.update(metrics)
    run.summary.update({"events_csv": str(events_csv), "hard_negative_dir": str(hard_negative_dir)})
    run.log(metrics)

    artifact = wandb.Artifact("hard-negative-dataset", type="dataset")
    events_path = Path(events_csv)
    if events_path.exists():
        artifact.add_file(str(events_path))
    hardneg_path = Path(hard_negative_dir)
    if hardneg_path.exists():
        artifact.add_dir(str(hardneg_path), name="yolo")
    run.log_artifact(artifact, aliases=["latest"])

    url = run.get_url()
    run.finish()
    return url


def log_yolo_retrain_run(
    project: str,
    candidate_weights: str | Path | None,
    combined_yaml: str | Path,
    base_weights: str | Path,
    hard_negative_dir: str | Path,
    epochs: int,
    device: str | int,
) -> str | None:
    """Log a YOLO hard-negative fine-tuning run before eval-gate promotion."""
    try:
        import wandb
    except ModuleNotFoundError:
        return None

    os.environ.setdefault("WANDB_MODE", "offline")
    run = wandb.init(project=project, job_type="yolo-retrain", reinit=True)
    metrics = _count_yolo_dataset(hard_negative_dir)
    run.summary.update(metrics)
    run.summary.update(
        {
            "candidate_weights": str(candidate_weights) if candidate_weights else None,
            "combined_yaml": str(combined_yaml),
            "base_weights": str(base_weights),
            "hard_negative_dir": str(hard_negative_dir),
            "epochs": epochs,
            "device": str(device),
        }
    )
    run.log(metrics)

    if candidate_weights and Path(candidate_weights).exists():
        artifact = wandb.Artifact(Path(candidate_weights).stem, type="model")
        artifact.add_file(str(candidate_weights))
        run.log_artifact(artifact, aliases=["latest", "candidate"])

    url = run.get_url()
    run.finish()
    return url


def log_detection_gate_run(
    project: str,
    result: dict[str, Any],
    registry_dir: str | Path,
    hard_negative_dir: str | Path | None = None,
) -> str | None:
    """Log a YOLO detection gate decision as a W&B run + model artifact.

    The real-data track stores YOLO checkpoints in ``LocalModelRegistry``. This
    logger mirrors the synthetic webhook logging for detection metrics: candidate
    and production eval numbers, per-check pass/fail, the promotion decision, and
    the registered ``.pt`` model artifact. Returns the run URL, or ``None`` when
    W&B is unavailable.
    """
    try:
        import wandb
    except ModuleNotFoundError:
        return None

    os.environ.setdefault("WANDB_MODE", "offline")
    run = wandb.init(project=project, job_type="eval-gate", reinit=True)
    metrics = {
        **_flatten_metrics("candidate", result.get("candidate_metrics")),
        **_flatten_metrics("production", result.get("production_metrics")),
        "promoted": int(bool(result.get("promoted"))),
    }
    run.summary.update(metrics)
    run.summary.update(
        {
            "version": result.get("version"),
            "gate_reason": result.get("reason"),
            "candidate_weights": result.get("candidate_weights"),
            "production_weights": result.get("production_weights"),
        }
    )
    run.log(metrics)

    checks = result.get("checks") or {}
    if checks:
        table = wandb.Table(
            columns=["check", "passed", "candidate", "production", "detail"]
        )
        for name, check in sorted(checks.items()):
            table.add_data(
                name,
                bool(check.get("passed")),
                check.get("candidate"),
                check.get("production"),
                check.get("detail"),
            )
        run.log({"gate_checks": table})

    version = result.get("version")
    model_path = None
    if version:
        model_dir = Path(registry_dir) / "models" / str(version)
        model_path = model_dir / "model.pt"
        if not model_path.exists():
            candidates = sorted(model_dir.glob("*.pt"))
            model_path = candidates[0] if candidates else None
    if version and model_path and model_path.exists():
        artifact = wandb.Artifact(str(version), type="model")
        artifact.add_file(str(model_path))
        aliases = ["latest", "production"] if result.get("promoted") else ["latest", "candidate"]
        run.log_artifact(artifact, aliases=aliases)

    hardneg_path = Path(hard_negative_dir) if hard_negative_dir else None
    if hardneg_path and hardneg_path.exists():
        artifact = wandb.Artifact(f"{version or 'candidate'}-hard-negatives", type="dataset")
        artifact.add_dir(str(hardneg_path))
        run.log_artifact(artifact, aliases=["latest"])

    url = run.get_url()
    run.finish()
    return url
