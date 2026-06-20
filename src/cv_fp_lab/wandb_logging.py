from __future__ import annotations

from pathlib import Path

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


def log_retrain_run(project: str, result: dict, registry_dir: str | Path) -> str | None:
    """Log a webhook-triggered retrain as a W&B run with metrics + model artifact.

    Returns the run URL, or None if wandb is unavailable. Honors WANDB_MODE, so it
    is a no-op-to-disk in offline mode and syncs to the server when online.
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
