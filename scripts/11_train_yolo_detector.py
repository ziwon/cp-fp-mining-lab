from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cv_fp_lab.config import load_config
from cv_fp_lab.yolo_detector import YoloDetector


def main() -> None:
    cfg = load_config()
    y = cfg["yolo"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=y["epochs"])
    parser.add_argument("--device", default=str(y["device"]), help="cpu or GPU index, e.g. 0")
    parser.add_argument("--base-weights", default=y["base_weights"])
    args = parser.parse_args()

    data_yaml = Path(cfg["dfire"]["data_yaml"])
    if not data_yaml.exists():
        raise FileNotFoundError("Run scripts/10_prepare_dfire.py first to create the dataset YAML.")

    detector = YoloDetector.train(
        data_yaml=data_yaml,
        base_weights=args.base_weights,
        epochs=args.epochs,
        imgsz=y["imgsz"],
        batch=y["batch"],
        device=args.device,
        project=y["runs_dir"],
        name=y["run_name"],
    )
    print(f"Trained YOLO detector. Best weights: {detector.weights_path}")


if __name__ == "__main__":
    main()
