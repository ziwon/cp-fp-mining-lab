from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cv_fp_lab.config import load_config
from cv_fp_lab.yolo_detector import write_dfire_yaml


def main() -> None:
    cfg = load_config()
    d = cfg["dfire"]
    for key in ("train_dir", "val_dir"):
        p = Path(d[key]) / "images"
        if not p.exists():
            raise FileNotFoundError(f"{key} images not found: {p}")
    out = write_dfire_yaml(
        d["data_yaml"],
        train_dir=d["train_dir"],
        val_dir=d["val_dir"],
        classes={int(k): v for k, v in d["classes"].items()},
    )
    print(f"Wrote dataset YAML: {out}")
    print(f"  train: {Path(d['train_dir']) / 'images'}")
    print(f"  val:   {Path(d['val_dir']) / 'images'}")


if __name__ == "__main__":
    main()
