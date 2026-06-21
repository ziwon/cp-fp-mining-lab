from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cv_fp_lab.config import load_config
from cv_fp_lab.dataset_builder import build_hard_negative_dataset
from cv_fp_lab.wandb_logging import log_hard_negative_dataset_run


def main() -> None:
    cfg = load_config()
    hn = cfg["hard_negatives"]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events",
        default=None,
        help=(
            "mined/reviewed FP events CSV (needs source_image_path). "
            "Defaults to reviewed_fp_samples.csv when present, else fp_events.csv."
        ),
    )
    args = parser.parse_args()

    processed_dir = Path(cfg["paths"]["processed_dir"])
    reviewed = processed_dir / "reviewed_fp_samples.csv"
    events = Path(args.events) if args.events else reviewed if reviewed.exists() else processed_dir / "fp_events.csv"
    n_classes = len(cfg["dfire"]["classes"])
    stats = build_hard_negative_dataset(events, hn["output_dir"], n_classes=n_classes)
    print(f"Built hard-negative dataset: {stats['output_dir']}")
    print(f"  source events: {events}")
    print(f"  images: {stats['n_images']} (negatives: {stats['n_negatives']})")
    print(f"  invalid labels dropped: {stats['n_invalid_labels_dropped']}")
    try:
        wandb_url = log_hard_negative_dataset_run(
            project=cfg["wandb"]["project"],
            events_csv=events,
            hard_negative_dir=hn["output_dir"],
            stats=stats,
        )
        if wandb_url:
            print(f"W&B hard-negative run: {wandb_url}")
    except Exception as exc:
        print(f"W&B logging skipped: {exc}")
    print("Next: scripts/15_retrain_with_hard_negatives.py")


if __name__ == "__main__":
    main()
