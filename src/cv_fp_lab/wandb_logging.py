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
