# Architecture

## Goal

Build a data-centric feedback loop for production computer vision systems that suffer from repeated false positives.

## Pipeline

```text
Production Inference
  event_id, camera_id, timestamp, bbox, class, confidence, model_version
        ↓
FP/FN Candidate Mining
  false positive, false negative, low confidence, temporal flicker, new camera/site
        ↓
Frame + Crop Store
  full_frame.jpg, bbox_crop.jpg, short_clip.mp4, metadata.json
        ↓
Embedding + Clustering
  CLIP/DINOv2/simple embeddings → UMAP → HDBSCAN/K-Means
        ↓
Human Review
  Label Studio: event validity, fp_type, bbox_valid, comment
        ↓
Dataset Builder
  Label Studio JSON → COCO/YOLO hard-negative dataset
        ↓
W&B
  Artifacts: dataset versions
  Tables: FP/FN visual comparison
  Runs: training/evaluation metrics
  Registry: candidate/staging/production model
```

## GPU server deployment

The first deployable shape is a single dedicated GPU server running Docker Compose:

```text
GPU Server
  Docker Compose
    miner
      Python worker image
      embedding extraction, clustering, dataset export/import, W&B logging
      /app/data mounted from host storage
      optional NVIDIA GPU access for CLIP/DINOv2 embeddings

    MinIO
      S3-compatible frame/crop/object store
      local stand-in for cloud object storage

    Label Studio
      review UI for event validity, fp_type, bbox quality, comments

    W&B
      offline local run directory by default
      online project/artifact store in production
```

This keeps the mining logic stateless. The worker reads config, event metadata, and object paths; writes versioned outputs; and exits. Long-lived state belongs in object storage, Label Studio, W&B, or an external metadata database.

## Kubernetes and cloud scalability

The Compose services map directly to cloud-native components:

| Compose service | Kubernetes / cloud equivalent |
| --- | --- |
| `miner` | `Job`, `CronJob`, Argo Workflow step, or Airflow KubernetesPodOperator |
| local `data/` bind mount | S3, GCS, Azure Blob, MinIO, EFS, or a PVC |
| `minio` | Managed object storage or in-cluster MinIO |
| `labelstudio` | Deployment + managed PostgreSQL + object storage |
| local W&B offline logs | W&B SaaS/on-prem Artifacts, Tables, Runs, Registry |

Scale-out boundaries:

- **Batch partitioning**: split mining jobs by date range, site, camera group, model version, or event class.
- **GPU pools**: schedule embedding jobs on GPU nodes; run clustering, export, and import on CPU nodes.
- **Object storage contract**: store images, crops, clips, embeddings, and dataset exports behind URI-based paths instead of local-only paths.
- **Metadata contract**: move event metadata from CSV to PostgreSQL, BigQuery, Snowflake, or Kafka-derived parquet tables.
- **Workflow orchestration**: run nightly or hourly mining through Argo Workflows, Airflow, Prefect, or a cloud batch service.
- **Review loop**: keep Label Studio asynchronous; mining can export tasks and continue while reviewers work.
- **Lineage**: publish every curated dataset as a W&B Artifact with source query, model version, config hash, and review export version.

Recommended production progression:

1. **Single GPU server**: Docker Compose, host-mounted data, MinIO, Label Studio, offline/online W&B.
2. **Shared storage**: replace host paths with object storage and keep Compose for the worker.
3. **Scheduled jobs**: run the same image as a Kubernetes `CronJob` or workflow step.
4. **Elastic mining**: shard embedding extraction across GPU jobs and merge embeddings before clustering.
5. **Managed platform**: managed object storage, managed PostgreSQL, cloud secrets, observability, and model/data registry.

For the production hybrid Kubernetes target architecture, see [`docs/hybrid_k8s_architecture.md`](hybrid_k8s_architecture.md).

## Production extension points

- Replace synthetic images with S3/MinIO image paths.
- Add PostgreSQL/Kafka event extraction.
- Generate bbox crops as separate artifacts.
- Use DINOv2 or CLIP embeddings.
- Use FiftyOne for visual EDA.
- Trigger this pipeline nightly via Argo Workflows or Airflow.
