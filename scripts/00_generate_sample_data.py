from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cv_fp_lab.config import load_config
from cv_fp_lab.dataset import generate_sample_data
from cv_fp_lab.utils import ensure_dir


def main() -> None:
    cfg = load_config()
    raw_dir = cfg["paths"]["raw_dir"]
    processed_dir = ensure_dir(cfg["paths"]["processed_dir"])
    sample = cfg["sample_data"]
    df = generate_sample_data(
        raw_dir=raw_dir,
        fp_types=sample["fp_types"],
        samples_per_type=sample["samples_per_type"],
        image_size=sample["image_size"],
        seed=sample["seed"],
    )
    out = processed_dir / "fp_events.csv"
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} synthetic FP samples")
    print(f"Metadata: {out}")


if __name__ == "__main__":
    main()
