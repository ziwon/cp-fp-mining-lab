# Real-Data Track — D-Fire detector + false-positive mining

The default pipeline mines **synthetic** false positives. This track replaces the
synthetic generator (`scripts/00`) with a real loop on the
[D-Fire](https://github.com/gaiasd/DFireDataset) fire/smoke detection dataset:
train a real detector, run it, and harvest its **actual** false positives as the
mining input. Everything downstream (embeddings → cluster → review → W&B) consumes
the same `fp_events.csv` schema unchanged.

This is steps 1–2 of the production-hardening plan in
[`active_learning.md`](active_learning.md): a real detector and real FP collection.

## Install

```bash
uv sync --extra detect          # adds ultralytics (pulls torch)
```

CPU works for small subsets. For GPU, set `yolo.device: 0` (and
`embedding.device: cuda`) in `configs/pipeline.yaml`.

### GPU setup (NVIDIA Blackwell, RTX 50xx / sm_120)

Blackwell needs a recent CUDA torch build (the default PyPI torch fails with a
sm_120 kernel mismatch). With driver 580+ (CUDA 13 capable), a dedicated GPU venv:

```bash
uv venv .venv-gpu --python 3.10
uv pip install --python .venv-gpu/bin/python --index-strategy unsafe-best-match \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  torch torchvision -e . ultralytics "transformers>=4.40,<5"
# verify
.venv-gpu/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Verified on an RTX 5080: torch 2.12.1+cu130, `cuda available: True`, capability
(12, 0); CLIP embeds ~35 img/s and YOLO predicts on `device=0`. On this host the
GPU venv lives at `/data/venvs/cpfp-gpu` (small workspace disk).

## Configure

The `dfire`, `yolo`, and `mining` blocks in `configs/pipeline.yaml` point at the
dataset and output locations. Defaults use the 300-image `dfire-300-stratified`
subset for fast iteration; switch `dfire.train_dir` to the full `D-Fire/train` for
real training. Heavy outputs (runs, weights, crops) default to `/data/...` because
the workspace disk is small on this host.

## Run

```bash
just detect-prepare     # write the Ultralytics dataset YAML (0=smoke, 1=fire)
just detect-train       # fine-tune yolov8n on D-Fire -> best.pt
just mine-fp            # run the detector over a split, harvest FP crops
# then the normal pipeline on the real FPs:
uv run python scripts/01_extract_embeddings.py --method simple
uv run python scripts/02_cluster_false_positives.py
uv run python scripts/03_export_for_label_studio.py
```

`mine-fp` writes `data/processed/fp_events.csv` (same schema as the synthetic
generator, plus `source_image_path` and `is_negative_image`) and the FP crops.

## How a false positive is defined

A detection is a false positive when it matches **no ground-truth box** — same
class and IoU ≥ `mining.iou_thr` is a match (`fp_mining.is_false_positive`). Two
sources dominate:

- **Negative images** (empty YOLO label = no fire/smoke): any detection is an FP.
  D-Fire's train split has ~7,800 such images — the richest FP source.
- **Misclassification / mislocalization** on positive images: e.g. predicting
  `fire` over a `smoke` region, or a box that misses the object.

The crop of each FP region becomes a review task — the real analogue of the
synthetic steam/reflection/headlight samples.

## Notes & next steps

- An **underfit** detector (few epochs) yields *more* FPs — useful for exercising
  the loop, but use a properly trained detector and a sane `mining.conf` for real
  curation.
- `simple` embeddings (color/edge histograms) cannot separate real FP types well;
  clusters collapse. **Step 3 (done)** implements the `clip` path
  (`uv sync --extra clip`, then `--method clip` on script 01): on the 244 mined
  D-Fire FPs, CLIP took HDBSCAN from 2 collapsed clusters to 9, with fire FPs
  separating into their own clusters. (Euclidean silhouette undersells CLIP since
  it lives in cosine space — judge by cluster structure, not euclidean distance.)
- **Step 4**: gate retraining on a frozen eval set (D-Fire `test` + the
  `verifier-eval` video clips) with mAP / per-class recall / FP-rate, not the
  current hold-out accuracy.
