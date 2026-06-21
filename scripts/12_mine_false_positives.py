from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cv_fp_lab.config import load_config
from cv_fp_lab.fp_mining import collect_false_positives
from cv_fp_lab.utils import ensure_dir
from cv_fp_lab.wandb_logging import log_mining_run
from cv_fp_lab.yolo_detector import YoloDetector


def main() -> None:
    cfg = load_config()
    m = cfg["mining"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=cfg["yolo"]["weights"])
    parser.add_argument("--limit", type=int, default=m["limit"])
    parser.add_argument("--conf", type=float, default=m["conf"])
    parser.add_argument("--iou", type=float, default=m["iou_thr"])
    parser.add_argument("--images", default=m["source_images"], help="override source images dir")
    parser.add_argument("--labels", default=m["source_labels"], help="override source labels dir")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"No detector weights at {weights}. Train one (script 11) first.")

    detector = YoloDetector.load(weights)
    df = collect_false_positives(
        detector,
        images_dir=args.images,
        labels_dir=args.labels,
        crops_dir=m["crops_dir"],
        model_version=weights.stem if weights.stem != "best" else f"yolo-{weights.parent.parent.name}",
        conf=args.conf,
        iou_thr=args.iou,
        limit=args.limit,
    )

    processed_dir = ensure_dir(cfg["paths"]["processed_dir"])
    out = processed_dir / "fp_events.csv"
    df.to_csv(out, index=False)
    n_neg = int(df["is_negative_image"].sum()) if len(df) else 0
    print(f"Mined {len(df)} false positives -> {out}")
    if len(df):
        print(f"  from negative images: {n_neg} | from mislocalized/misclassified: {len(df) - n_neg}")
        print(f"  by predicted class:\n{df['pred_class'].value_counts().to_string()}")
    print(f"  crops: {cfg['mining']['crops_dir']}")
    try:
        wandb_url = log_mining_run(
            project=cfg["wandb"]["project"],
            events_csv=out,
            crops_dir=m["crops_dir"],
            weights=weights,
            source_images=args.images,
            source_labels=args.labels,
            conf=args.conf,
            iou_thr=args.iou,
            limit=args.limit,
        )
        if wandb_url:
            print(f"W&B mining run: {wandb_url}")
    except Exception as exc:
        print(f"W&B logging skipped: {exc}")
    print("Next: scripts/01_extract_embeddings.py -> 02_cluster -> 03_export_for_label_studio")


if __name__ == "__main__":
    main()
