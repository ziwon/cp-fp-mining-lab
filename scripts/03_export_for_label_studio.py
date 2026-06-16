from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cv_fp_lab.config import load_config
from cv_fp_lab.labelstudio import LABEL_CONFIG, dataframe_to_labelstudio_tasks
from cv_fp_lab.utils import ensure_dir


def main() -> None:
    cfg = load_config()
    processed_dir = ensure_dir(cfg["paths"]["processed_dir"])
    df = pd.read_csv(processed_dir / "fp_clusters.csv")
    output_path = processed_dir / "labelstudio_tasks.json"
    tasks = dataframe_to_labelstudio_tasks(df, output_path)
    (processed_dir / "labelstudio_label_config.xml").write_text(LABEL_CONFIG, encoding="utf-8")
    print(f"Exported {len(tasks)} Label Studio tasks: {output_path}")
    print(f"Label config: {processed_dir / 'labelstudio_label_config.xml'}")


if __name__ == "__main__":
    main()
