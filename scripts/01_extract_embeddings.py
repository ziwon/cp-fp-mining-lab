from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cv_fp_lab.config import load_config
from cv_fp_lab.embeddings import extract_embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=None, choices=["simple", "clip"])
    args = parser.parse_args()

    cfg = load_config()
    method = args.method or cfg["embedding"]["method"]
    metadata_file = Path(cfg["embedding"]["metadata_file"])
    if not metadata_file.exists():
        raise FileNotFoundError("Run scripts/00_generate_sample_data.py first")

    df = pd.read_csv(metadata_file)
    embeddings = extract_embeddings(df["image_path"].tolist(), method=method)
    out = Path(cfg["embedding"]["output_file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, embeddings)
    print(f"Saved embeddings: {out} shape={embeddings.shape}")


if __name__ == "__main__":
    main()
