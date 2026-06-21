from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_dir


def _label_path_for(image_path: str | Path) -> Path:
    """Derive the YOLO label path from an image path (.../images/x.jpg -> .../labels/x.txt)."""
    p = Path(image_path)
    parts = list(p.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _unique_stem(src: Path, used_stems: set[str]) -> str:
    if src.stem not in used_stems:
        used_stems.add(src.stem)
        return src.stem
    digest = hashlib.sha1(str(src.resolve()).encode("utf-8")).hexdigest()[:8]
    stem = f"{src.stem}_{digest}"
    used_stems.add(stem)
    return stem


def validate_label_lines(label_path: str | Path, n_classes: int) -> tuple[list[str], int]:
    """Validate YOLO label lines; return (valid lines, count of dropped invalid).

    A line is valid when it is ``class cx cy w h`` with an in-range integer class
    and four normalized floats in [0, 1]. This is the dataset-validation guard
    that keeps corrupt/out-of-bounds labels (D-Fire has a few) out of retraining.
    """
    path = Path(label_path)
    if not path.exists():
        return [], 0
    valid: list[str] = []
    invalid = 0
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            if line.strip():
                invalid += 1
            continue
        try:
            cls = int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            invalid += 1
            continue
        if not (0 <= cls < n_classes) or not all(0.0 <= c <= 1.0 for c in coords):
            invalid += 1
            continue
        valid.append(f"{cls} {' '.join(parts[1:])}")
    return valid, invalid


def build_hard_negative_dataset(
    events: pd.DataFrame | str | Path,
    output_dir: str | Path,
    n_classes: int = 2,
    confirmed_only: bool = True,
) -> dict[str, Any]:
    """Build a YOLO hard-negative dataset from confirmed false-positive events.

    For each unique ``source_image_path`` of a confirmed FP, copy the full source
    frame and write its **validated ground-truth** label (the true objects, if
    any) — so the detector relearns the correct content of FP-prone images.
    Negative source images get an empty label (pure background hard negatives).

    ``confirmed_only`` keeps events the reviewer (or ground truth) marked
    ``false_positive`` — in ``operator_feedback`` or ``review_is_event``.
    """
    if isinstance(events, (str, Path)):
        events = pd.read_csv(events)

    df = events
    if confirmed_only:
        confirmed = pd.Series(False, index=df.index)
        if "operator_feedback" in df.columns:
            confirmed |= df["operator_feedback"].astype(str).eq("false_positive")
        if "review_is_event" in df.columns:
            confirmed |= df["review_is_event"].astype(str).eq("false_positive")
        df = df[confirmed]

    if "source_image_path" not in df.columns:
        raise ValueError("events need a 'source_image_path' column (mined FP schema).")

    out = ensure_dir(output_dir)
    img_dir = ensure_dir(out / "images")
    lbl_dir = ensure_dir(out / "labels")

    n_images = 0
    n_negatives = 0
    n_invalid_dropped = 0
    used_stems: set[str] = set()
    for src in sorted({s for s in df["source_image_path"].dropna() if Path(s).exists()}):
        src = Path(src)
        out_stem = _unique_stem(src, used_stems)
        shutil.copy2(src, img_dir / f"{out_stem}{src.suffix}")
        valid_lines, invalid = validate_label_lines(_label_path_for(src), n_classes)
        n_invalid_dropped += invalid
        (lbl_dir / f"{out_stem}.txt").write_text("\n".join(valid_lines), encoding="utf-8")
        n_images += 1
        if not valid_lines:
            n_negatives += 1

    return {
        "output_dir": str(out),
        "n_images": n_images,
        "n_negatives": n_negatives,
        "n_invalid_labels_dropped": n_invalid_dropped,
    }
