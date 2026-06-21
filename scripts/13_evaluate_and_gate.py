from pathlib import Path
import argparse
import datetime as dt
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cv_fp_lab.config import load_config
from cv_fp_lab.detection_eval import detection_metrics, false_positive_rate
from cv_fp_lab.gating import evaluate_promotion
from cv_fp_lab.registry import LocalModelRegistry
from cv_fp_lab.yolo_detector import YoloDetector


def _evaluate(weights: Path, cfg: dict, device: str) -> dict:
    g = cfg["gate"]
    metrics = detection_metrics(
        weights, cfg["dfire"]["data_yaml"], split=g["eval_split"],
        conf=g["conf"], iou=g["iou"], device=device,
    )
    fp = false_positive_rate(
        YoloDetector.load(weights),
        images_dir=Path(cfg["dfire"]["val_dir"]) / "images",
        labels_dir=Path(cfg["dfire"]["val_dir"]) / "labels",
        conf=g["conf"], iou_thr=g["iou"], limit=g["fp_eval_limit"],
    )
    return {**metrics, **fp}


def main() -> None:
    cfg = load_config()
    g = cfg["gate"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default=cfg["yolo"]["weights"], help="candidate weights (.pt)")
    parser.add_argument("--device", default="cpu", help="cpu or GPU index, e.g. 0")
    args = parser.parse_args()

    candidate_weights = Path(args.candidate)
    if not candidate_weights.exists():
        raise FileNotFoundError(f"No candidate weights at {candidate_weights}")

    registry = LocalModelRegistry(g["registry_dir"])
    prod_weights = registry.weights_path("production")

    print(f"Evaluating candidate: {candidate_weights}")
    cand = _evaluate(candidate_weights, cfg, args.device)
    prod = None
    if prod_weights:
        print(f"Evaluating production: {prod_weights}")
        prod = _evaluate(prod_weights, cfg, args.device)

    decision = evaluate_promotion(
        cand, prod,
        map50_min_delta=g["map50_min_delta"],
        recall_min_delta=g["recall_min_delta"],
        fp_rate_max_delta=g["fp_rate_max_delta"],
    )

    print("\n=== candidate metrics ===")
    print(f"  mAP@50={cand['map50']:.4f}  recall={cand['recall']:.4f}  "
          f"neg_fp_rate={cand['neg_fp_rate']:.4f}  mean_fp/img={cand['mean_fp_per_image']:.3f}")
    print("\n=== gate decision ===")
    for name, c in decision["checks"].items():
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {name}: {c['detail']}")
    print(f"  -> promoted={decision['promoted']} ({decision['reason']})")

    version = f"detector-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    registry.register_file(version, candidate_weights, metrics=cand, stage="candidate",
                           extra_meta={"gate": decision["reason"]})
    if decision["promoted"]:
        registry.promote(version, "staging")
        registry.promote(version, "production")
        print(f"\nPromoted {version} -> production")
    else:
        print(f"\nRegistered {version} as candidate (gate not cleared; production unchanged)")


if __name__ == "__main__":
    main()
