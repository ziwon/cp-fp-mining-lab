from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from .utils import ensure_dir


FP_COLORS = {
    "steam": (210, 210, 210),
    "fog": (185, 190, 195),
    "reflection": (255, 230, 120),
    "headlight": (255, 245, 180),
    "shadow": (50, 50, 60),
    "animal": (120, 85, 50),
}

PRED_CLASS_BY_FP = {
    "steam": "smoke",
    "fog": "smoke",
    "reflection": "fire",
    "headlight": "fire",
    "shadow": "falldown",
    "animal": "intrusion",
}

# Per-FP-type detector confidence ranges. Ambiguous classes (reflection, shadow)
# sit near the decision boundary so uncertainty-based ranking surfaces them first;
# clear-cut classes (steam, fog) score high and confident.
CONFIDENCE_BY_FP = {
    "steam": (0.80, 0.96),
    "fog": (0.78, 0.95),
    "headlight": (0.60, 0.80),
    "animal": (0.62, 0.82),
    "reflection": (0.52, 0.70),
    "shadow": (0.50, 0.68),
}


def _draw_pattern(draw: ImageDraw.ImageDraw, fp_type: str, size: int, rng: random.Random) -> tuple[int, int, int, int]:
    color = FP_COLORS[fp_type]
    if fp_type in {"steam", "fog"}:
        x0 = rng.randint(35, 90)
        y0 = rng.randint(45, 120)
        x1 = rng.randint(130, 200)
        y1 = rng.randint(130, 210)
        for _ in range(10):
            ox = rng.randint(-25, 25)
            oy = rng.randint(-25, 25)
            draw.ellipse((x0 + ox, y0 + oy, x1 + ox, y1 + oy), fill=color)
        return x0, y0, x1, y1
    if fp_type in {"reflection", "headlight"}:
        x0 = rng.randint(30, 110)
        y0 = rng.randint(40, 140)
        x1 = x0 + rng.randint(45, 90)
        y1 = y0 + rng.randint(25, 65)
        draw.ellipse((x0, y0, x1, y1), fill=color)
        draw.line((x0, y0, x1 + 50, y1 + 15), fill=color, width=4)
        return x0, y0, min(size - 1, x1 + 50), min(size - 1, y1 + 15)
    if fp_type == "shadow":
        x0 = rng.randint(40, 90)
        y0 = rng.randint(100, 155)
        x1 = rng.randint(130, 200)
        y1 = rng.randint(145, 210)
        draw.polygon([(x0, y1), (x1, y0), (x1, y1)], fill=color)
        return x0, y0, x1, y1
    # animal-like blob
    x0 = rng.randint(45, 115)
    y0 = rng.randint(110, 165)
    x1 = x0 + rng.randint(55, 90)
    y1 = y0 + rng.randint(25, 45)
    draw.ellipse((x0, y0, x1, y1), fill=color)
    draw.ellipse((x1 - 15, y0 - 10, x1 + 20, y0 + 20), fill=color)
    return x0, max(0, y0 - 10), min(size - 1, x1 + 20), y1


def generate_sample_data(raw_dir: str | Path, fp_types: list[str], samples_per_type: int, image_size: int, seed: int) -> pd.DataFrame:
    raw_dir = ensure_dir(raw_dir)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    rows = []

    for fp_type in fp_types:
        for i in range(samples_per_type):
            bg = np_rng.normal(loc=75, scale=12, size=(image_size, image_size, 3)).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(bg, mode="RGB")
            draw = ImageDraw.Draw(img, "RGBA")
            bbox = _draw_pattern(draw, fp_type, image_size, rng)
            if fp_type in {"steam", "fog"}:
                img = img.filter(ImageFilter.GaussianBlur(radius=2.2 if fp_type == "fog" else 1.2))
            event_id = f"evt_{fp_type}_{i:04d}"
            filename = f"{event_id}.png"
            img.save(raw_dir / filename)
            rows.append(
                {
                    "event_id": event_id,
                    "image_path": str(raw_dir / filename),
                    "camera_id": f"cam-{rng.randint(1, 8):03d}",
                    "site_id": f"site-{rng.randint(1, 3):02d}",
                    "timestamp": f"2026-06-{rng.randint(1, 15):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00+09:00",
                    "pred_class": PRED_CLASS_BY_FP[fp_type],
                    "pred_confidence": round(rng.uniform(*CONFIDENCE_BY_FP[fp_type]), 3),
                    "bbox_x0": bbox[0],
                    "bbox_y0": bbox[1],
                    "bbox_x1": bbox[2],
                    "bbox_y1": bbox[3],
                    "operator_feedback": "false_positive",
                    "synthetic_fp_type": fp_type,
                    "model_version": "detector-v0.synthetic",
                }
            )
    return pd.DataFrame(rows)
