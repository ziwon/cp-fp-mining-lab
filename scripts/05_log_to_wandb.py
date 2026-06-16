from pathlib import Path
import os
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cv_fp_lab.config import load_config
from cv_fp_lab.wandb_logging import log_table_and_artifact


def main() -> None:
    cfg = load_config()
    os.environ.setdefault("WANDB_MODE", "offline")
    project = os.getenv("WANDB_PROJECT", cfg["wandb"]["project"])
    log_table_and_artifact(
        project=project,
        artifact_name=cfg["wandb"]["artifact_name"],
        processed_dir=cfg["paths"]["processed_dir"],
    )
    print("Logged FP table and dataset artifact to W&B")


if __name__ == "__main__":
    main()
