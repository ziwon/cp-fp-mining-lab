from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def simple_image_embedding(image_path: str | Path) -> np.ndarray:
    """Small, deterministic image descriptor for offline demos.

    Features:
    - RGB histogram
    - grayscale histogram
    - simple edge intensity summary
    - low-resolution thumbnail pixels
    """
    img = Image.open(image_path).convert("RGB").resize((64, 64))
    arr = np.asarray(img).astype(np.float32) / 255.0
    feats: list[np.ndarray] = []
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=16, range=(0.0, 1.0), density=True)
        feats.append(hist.astype(np.float32))
    gray = arr.mean(axis=2)
    ghist, _ = np.histogram(gray, bins=16, range=(0.0, 1.0), density=True)
    feats.append(ghist.astype(np.float32))
    gy, gx = np.gradient(gray)
    edge = np.sqrt(gx**2 + gy**2)
    feats.append(np.array([edge.mean(), edge.std(), gray.mean(), gray.std()], dtype=np.float32))
    thumb = np.asarray(img.resize((8, 8))).astype(np.float32).reshape(-1) / 255.0
    feats.append(thumb)
    vec = np.concatenate(feats).astype(np.float32)
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm


def extract_embeddings(image_paths: list[str | Path], method: str = "simple") -> np.ndarray:
    if method == "simple":
        return np.vstack([simple_image_embedding(p) for p in image_paths])
    if method == "clip":
        raise NotImplementedError(
            "CLIP mode is intentionally left as an extension point. "
            "Install torch/transformers and implement CLIP image embeddings here."
        )
    raise ValueError(f"Unsupported embedding method: {method}")
