from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cv_fp_lab.config import load_config
from cv_fp_lab.labelstudio import parse_labelstudio_export
from cv_fp_lab.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Label Studio export JSON")
    args = parser.parse_args()

    cfg = load_config()
    processed_dir = ensure_dir(cfg["paths"]["processed_dir"])
    df = parse_labelstudio_export(args.input)
    out = processed_dir / "reviewed_fp_samples.csv"
    df.to_csv(out, index=False)
    print(f"Imported reviewed samples: {out}")
    print(df.head())


if __name__ == "__main__":
    main()
