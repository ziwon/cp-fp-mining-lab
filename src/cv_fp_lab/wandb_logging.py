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
