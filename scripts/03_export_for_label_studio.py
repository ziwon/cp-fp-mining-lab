from pathlib import Path
import argparse
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cv_fp_lab.active_learning import STRATEGIES, select_for_review
from cv_fp_lab.config import load_config
from cv_fp_lab.labelstudio import LABEL_CONFIG, dataframe_to_labelstudio_tasks
from cv_fp_lab.utils import ensure_dir


def main() -> None:
    cfg = load_config()
    al = cfg.get("active_learning", {})
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--budget",
        type=int,
        default=al.get("budget"),
        help="Export only the top-N most informative tasks (default: all).",
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default=al.get("strategy", "least_confidence"),
        help="Acquisition function for uncertainty ranking.",
    )
    parser.add_argument(
        "--no-diversity",
        dest="diversity",
        action="store_false",
        help="Disable round-robin cluster diversity (plain uncertainty sort).",
    )
    parser.set_defaults(diversity=al.get("diversity", True))
    args = parser.parse_args()

    processed_dir = ensure_dir(cfg["paths"]["processed_dir"])
    df = pd.read_csv(processed_dir / "fp_clusters.csv")

    ranked = select_for_review(
        df,
        budget=args.budget,
        strategy=args.strategy,
        diversity=args.diversity,
    )
    ranking_path = processed_dir / "labelstudio_ranking.csv"
    ranked[
        ["event_id", "cluster_id", "pred_confidence", "uncertainty", "acquisition_rank"]
    ].to_csv(ranking_path, index=False)

    lscfg = cfg.get("labelstudio", {})
    image_mode = lscfg.get("image_mode", "path")
    if image_mode == "http":
        image_url_prefix = lscfg.get("http_base_url", "").rstrip("/") + "/"
    elif image_mode == "local_files":
        image_url_prefix = lscfg.get("local_files_url_prefix")
    else:
        image_url_prefix = None
    output_path = processed_dir / "labelstudio_tasks.json"
    tasks = dataframe_to_labelstudio_tasks(
        ranked,
        output_path,
        image_url_prefix=image_url_prefix,
        data_dir_strip=lscfg.get("data_dir_strip", ""),
    )
    (processed_dir / "labelstudio_label_config.xml").write_text(LABEL_CONFIG, encoding="utf-8")

    scope = f"top {len(tasks)} of {len(df)}" if args.budget else f"all {len(tasks)}"
    print(
        f"Exported {scope} Label Studio tasks "
        f"(strategy={args.strategy}, diversity={args.diversity})"
    )
    print(f"Tasks: {output_path}")
    print(f"Ranking: {ranking_path}")
    print(f"Label config: {processed_dir / 'labelstudio_label_config.xml'}")


if __name__ == "__main__":
    main()
