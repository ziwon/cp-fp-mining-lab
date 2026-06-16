# CV False Positive Mining Lab

A small, practical lab repository for experimenting with a computer-vision false-positive improvement loop:

```text
Production FP events
→ frame/crop collection
→ embedding extraction
→ UMAP/HDBSCAN clustering
→ human review / Label Studio export
→ YOLO/COCO-style dataset build
→ W&B Artifacts + Tables logging
→ retraining/evaluation handoff
```

This repository is intentionally lightweight. It uses a synthetic sample dataset so the pipeline can run without real CCTV footage.

## Why this exists

In production CV systems, false positives are not just failures. They are high-value signals showing where the model confuses real-world edge cases:

- smoke vs steam/fog/dust
- fire vs reflection/headlight/welding/sunset
- falldown vs sitting/crouching/shadow
- intrusion vs animal/tree motion/authorized worker

This lab demonstrates how to group those false positives, review them, and convert repeated patterns into hard-negative datasets.

## Repository layout

```text
cv-fp-mining-lab/
├── configs/
│   └── pipeline.yaml
├── data/
│   ├── raw/                  # generated sample frames/crops
│   ├── processed/            # embeddings, clusters, datasets
│   └── labelstudio_exports/  # sample Label Studio export JSON
├── docs/
│   ├── architecture.md
│   ├── fp_taxonomy.md
│   ├── hybrid_k8s_architecture.md
│   └── label_studio_setup.md
├── notebooks/
│   └── 01_fp_clustering.ipynb
├── scripts/
│   ├── 00_generate_sample_data.py
│   ├── 01_extract_embeddings.py
│   ├── 02_cluster_false_positives.py
│   ├── 03_export_for_label_studio.py
│   ├── 04_import_label_studio_export.py
│   └── 05_log_to_wandb.py
└── src/cv_fp_lab/
    ├── config.py
    ├── dataset.py
    ├── embeddings.py
    ├── clustering.py
    ├── labelstudio.py
    ├── wandb_logging.py
    └── utils.py
```

## Quick start

### Docker Compose on a GPU server

For a dedicated mining server, build the worker image and start the local services:

```bash
just docker-build
just docker-up
```

Run the full sample pipeline inside the container:

```bash
just docker-demo
```

Or open a shell in the GPU-enabled mining container:

```bash
just docker-shell
```

The Compose stack includes:

- `miner`: the reusable Python worker image for embedding, clustering, export, import, and W&B logging
- `minio`: local S3-compatible object storage for frame/crop artifacts
- `labelstudio`: human review UI at `http://localhost:8080`

On an NVIDIA GPU host, install the NVIDIA Container Toolkit and Docker Compose v2. The `miner` service requests all available GPUs through the Compose device reservation block.

### 1. Install dependencies with uv

```bash
uv sync
```

This creates `.venv/` and installs the locked runtime dependencies from `pyproject.toml` and `uv.lock`.

### 2. Generate synthetic false-positive samples

```bash
uv run python scripts/00_generate_sample_data.py
```

This creates small synthetic images representing patterns such as `steam`, `reflection`, `headlight`, `shadow`, and `animal`.

### 3. Extract embeddings

Default mode uses lightweight image features so the demo runs anywhere:

```bash
uv run python scripts/01_extract_embeddings.py --method simple
```

Optional CLIP mode is provided, but requires extra dependencies and model download:

```bash
uv sync --extra clip
uv run python scripts/01_extract_embeddings.py --method clip
```

### 4. Cluster false positives

```bash
uv run python scripts/02_cluster_false_positives.py
```

Outputs:

```text
data/processed/fp_clusters.csv
data/processed/fp_umap.csv
```

### 5. Export tasks for Label Studio

```bash
uv run python scripts/03_export_for_label_studio.py
```

Output:

```text
data/processed/labelstudio_tasks.json
```

### 6. Import reviewed Label Studio export

A small sample export is included:

```bash
uv run python scripts/04_import_label_studio_export.py \
  --input data/labelstudio_exports/sample_review_export.json
```

Output:

```text
data/processed/reviewed_fp_samples.csv
```

### 7. Log dataset and tables to W&B

Offline mode is enabled by default, so this works without a W&B account:

```bash
WANDB_MODE=offline uv run python scripts/05_log_to_wandb.py
```

To use your own W&B project:

```bash
wandb login
WANDB_MODE=online WANDB_PROJECT=cv-fp-mining-lab uv run python scripts/05_log_to_wandb.py
```

## Label Studio integration concept

This lab does not require a running Label Studio server, but the generated `labelstudio_tasks.json` follows a simple image classification/review task style. In a real deployment:

```text
S3/MinIO frame path
→ Label Studio task
→ human review: is_event, fp_type, bbox_valid
→ export JSON
→ dataset builder
→ W&B Artifact
```

See [`docs/label_studio_setup.md`](docs/label_studio_setup.md).

## W&B usage concept

W&B is used as the lineage and analysis layer:

- **Artifacts**: dataset versions such as `raw-fp-events:v1`, `hard-negative-smoke:v3`
- **Tables**: images, metadata, cluster IDs, human labels, model version
- **Runs**: training/evaluation metrics
- **Registry**: candidate/staging/production model lifecycle

## Suggested next steps

- Replace synthetic samples with real CCTV false-positive frames.
- Add production event metadata from PostgreSQL or Kafka.
- Use CLIP or DINOv2 embeddings instead of simple features.
- Add FiftyOne for visual error analysis.
- Add YOLO/RT-DETR training job and regression evaluation.
- Connect Label Studio webhooks to trigger dataset rebuilds.

## References to study

- Mindtech Global: false positive clustering for object detection data improvement
- ECCV 2018: unsupervised hard example mining from videos
- Label Studio YOLO ML backend
- W&B Artifacts / Tables / Ultralytics integration
- FiftyOne object detection evaluation
