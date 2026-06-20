from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cv_fp_lab.config import load_config
from cv_fp_lab.registry import LocalModelRegistry
from cv_fp_lab.training import retrain_and_register


def main() -> None:
    cfg = load_config()
    processed_dir = Path(cfg["paths"]["processed_dir"])
    dcfg = cfg["detector"]

    events_csv = Path(cfg["embedding"]["metadata_file"])
    embeddings_npy = Path(cfg["embedding"]["output_file"])
    reviewed_csv = processed_dir / "reviewed_fp_samples.csv"
    if not embeddings_npy.exists():
        raise FileNotFoundError("Run scripts 00-02 first to produce embeddings.")

    registry = LocalModelRegistry(dcfg["registry_dir"])
    result = retrain_and_register(
        registry,
        events_csv=events_csv,
        embeddings_npy=embeddings_npy,
        reviewed_csv=reviewed_csv if reviewed_csv.exists() else None,
        metric_key=dcfg["metric_key"],
        min_delta=dcfg["promotion_min_delta"],
    )
    print(f"Trained {result['version']} on {result['n_train']} samples")
    print(f"Metrics: {result['metrics']}")
    print(f"Promoted to production: {result['promoted']} ({result['reason']})")
    print(f"Production alias -> {registry.production_version()}")


if __name__ == "__main__":
    main()
