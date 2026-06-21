from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cv_fp_lab.config import load_config
from cv_fp_lab.dataset_builder import build_hard_negative_dataset


def main() -> None:
    cfg = load_config()
    hn = cfg["hard_negatives"]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events",
        default=str(Path(cfg["paths"]["processed_dir"]) / "fp_events.csv"),
        help="mined/reviewed FP events CSV (needs source_image_path)",
    )
    args = parser.parse_args()

    n_classes = len(cfg["dfire"]["classes"])
    stats = build_hard_negative_dataset(args.events, hn["output_dir"], n_classes=n_classes)
    print(f"Built hard-negative dataset: {stats['output_dir']}")
    print(f"  images: {stats['n_images']} (negatives: {stats['n_negatives']})")
    print(f"  invalid labels dropped: {stats['n_invalid_labels_dropped']}")
    print("Next: scripts/15_retrain_with_hard_negatives.py")


if __name__ == "__main__":
    main()
