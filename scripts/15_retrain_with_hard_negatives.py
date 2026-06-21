from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cv_fp_lab.config import load_config
from cv_fp_lab.yolo_detector import YoloDetector, write_dataset_yaml


def main() -> None:
    cfg = load_config()
    hn = cfg["hard_negatives"]
    y = cfg["yolo"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=hn["retrain_epochs"])
    parser.add_argument("--device", default=str(y["device"]))
    parser.add_argument(
        "--base-weights",
        default=cfg["gate"]["registry_dir"],
        help="fine-tune from the current production detector (registry) by default",
    )
    args = parser.parse_args()

    hardneg_dir = Path(hn["output_dir"])
    if not (hardneg_dir / "images").exists():
        raise FileNotFoundError("Run scripts/14_build_hard_negatives.py first.")

    # Train on [base D-Fire train + mined hard negatives], validate on the eval set.
    combined_yaml = write_dataset_yaml(
        hn["combined_yaml"],
        train_dirs=[hn["base_train_dir"], hardneg_dir],
        val_dir=cfg["dfire"]["val_dir"],
        classes={int(k): v for k, v in cfg["dfire"]["classes"].items()},
    )

    # Resolve the warm-start weights: production from the registry if available.
    from cv_fp_lab.registry import LocalModelRegistry

    base = args.base_weights
    if Path(base).is_dir() or base == cfg["gate"]["registry_dir"]:
        prod = LocalModelRegistry(cfg["gate"]["registry_dir"]).weights_path("production")
        base = str(prod) if prod else y["base_weights"]
    print(f"Warm-start weights: {base}")

    detector = YoloDetector.train(
        data_yaml=combined_yaml,
        base_weights=base,
        epochs=args.epochs,
        imgsz=y["imgsz"],
        batch=y["batch"],
        device=args.device,
        project=y["runs_dir"],
        name=hn["retrain_run"],
    )
    print(f"Retrained candidate: {detector.weights_path}")
    print("Next: scripts/13_evaluate_and_gate.py --candidate "
          f"{detector.weights_path} (gate decides promotion)")


if __name__ == "__main__":
    main()
