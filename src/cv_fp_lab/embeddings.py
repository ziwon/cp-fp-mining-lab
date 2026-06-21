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


_CLIP_CACHE: dict[str, tuple] = {}


def _load_clip(model_name: str, device: str):
    """Load and cache a transformers CLIP model + processor."""
    key = f"{model_name}@{device}"
    if key not in _CLIP_CACHE:
        try:
            import torch  # noqa: F401
            from transformers import CLIPModel, CLIPProcessor
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise SystemExit(
                "CLIP mode needs torch + transformers. Run `uv sync --extra clip`."
            ) from exc
        model = CLIPModel.from_pretrained(model_name).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_name)
        _CLIP_CACHE[key] = (model, processor)
    return _CLIP_CACHE[key]


def clip_image_embeddings(
    image_paths: list[str | Path],
    model_name: str = "openai/clip-vit-base-patch32",
    device: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """L2-normalized CLIP image embeddings — semantic features for FP clustering.

    Unlike the ``simple`` color/edge descriptor, CLIP separates real FP types
    (steam vs smoke, reflection vs fire) by appearance semantics. Returns an
    ``(N, D)`` float32 array; processes in batches and runs on GPU when available.
    """
    import torch

    if not image_paths:
        return np.empty((0, 512), dtype=np.float32)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = _load_clip(model_name, device)

    out: list[np.ndarray] = []
    for start in range(0, len(image_paths), batch_size):
        batch = image_paths[start : start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(out)


def extract_embeddings(
    image_paths: list[str | Path],
    method: str = "simple",
    clip_model: str = "openai/clip-vit-base-patch32",
    device: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    if method == "simple":
        return np.vstack([simple_image_embedding(p) for p in image_paths])
    if method == "clip":
        return clip_image_embeddings(image_paths, clip_model, device, batch_size)
    raise ValueError(f"Unsupported embedding method: {method}")
