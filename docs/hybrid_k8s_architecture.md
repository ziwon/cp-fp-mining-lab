# Hybrid Kubernetes Production Architecture

## Purpose

This document describes a production-grade architecture for running the false-positive mining loop across hybrid environments:

- **On-premises**: close to CCTV/video sources, GPU servers, private networks, and sensitive frame data.
- **Cloud**: elastic compute, managed storage, analytics, W&B, CI/CD, artifact registry, and long-term dataset governance.

The guiding principle is to keep the mining worker portable. The same container image should run as a Docker Compose service on one GPU server, a Kubernetes `Job` on an on-prem cluster, or a cloud batch/workflow step.

## Target Topology

```text
                         Cloud Region
  +--------------------------------------------------------------+
  |                                                              |
  |  Managed Container Registry                                  |
  |  Managed Object Storage / Dataset Lake                       |
  |  W&B SaaS or W&B Server                                      |
  |  CI/CD + GitOps Controller                                   |
  |  Optional Managed PostgreSQL / Analytics Warehouse           |
  |                                                              |
  +---------------------------^----------------------------------+
                              |
                    VPN / Direct Connect / PrivateLink
                              |
  +---------------------------v----------------------------------+
  |                    On-Premises Data Center                   |
  |                                                              |
  |  Kubernetes Cluster                                          |
  |    GPU Node Pool                                             |
  |      embedding Jobs: CLIP / DINOv2 / model-assisted mining   |
  |    CPU Node Pool                                             |
  |      clustering, export/import, dataset build, API workers   |
  |    Storage                                                   |
  |      MinIO / Ceph / NAS / CSI-backed PVCs                    |
  |    Review                                                    |
  |      Label Studio + PostgreSQL                               |
  |    Observability                                             |
  |      Prometheus, Grafana, Loki, OpenTelemetry                 |
  |                                                              |
  |  Production CV Systems                                       |
  |    cameras, inference services, event DB, Kafka topics       |
  |                                                              |
  +--------------------------------------------------------------+
```

## Core Components

| Component | On-premises role | Cloud role |
| --- | --- | --- |
| Mining worker image | Runs batch mining near private video data | Runs elastic reprocessing or backfills |
| Kubernetes Jobs | Execute embedding, clustering, export, import, dataset build | Same image and manifests for cloud batch |
| Object storage | MinIO, Ceph, NAS gateway, or S3-compatible appliance | S3, GCS, Azure Blob, or managed MinIO |
| Metadata store | PostgreSQL or Kafka-derived parquet tables | Managed PostgreSQL, BigQuery, Snowflake, Athena |
| Label Studio | Internal review UI for sensitive frames | Optional public/cloud review for sanitized datasets |
| W&B | Offline sync, on-prem W&B, or outbound SaaS sync | Primary lineage, Tables, Artifacts, Registry |
| Container registry | Pull-through cache or private registry mirror | Build and promote immutable mining images |
| Secrets | External Secrets + Vault/KMS | Cloud Secret Manager/KMS |
| Observability | Prometheus/Grafana/Loki on cluster | Cloud logs, metrics, tracing, alert routing |

## Namespace Layout

Use explicit namespaces so operations, permissions, and quotas stay clear:

```text
cv-mining-system
  controllers, service accounts, workflow templates, shared config

cv-mining-jobs
  short-lived Jobs/CronJobs for mining batches

cv-review
  Label Studio, PostgreSQL, review-facing ingress

cv-data
  MinIO/Ceph gateway, object-store operators, dataset maintenance jobs

cv-observability
  metrics, logs, dashboards, alerts
```

## Workload Model

### Scheduled mining

Nightly or hourly scheduled jobs discover new false-positive candidates and create review tasks.

```text
CronJob / Argo Workflow
  1. query event metadata
  2. resolve frame/crop object URIs
  3. run GPU embedding shards
  4. merge embeddings
  5. run UMAP + HDBSCAN/K-Means clustering
  6. export Label Studio tasks
  7. log W&B Tables and Artifacts
```

Recommended Kubernetes primitives:

- `CronJob` for simple scheduled runs.
- Argo Workflows for multi-step DAGs, fan-out/fan-in, retries, and artifact passing.
- Kueue or Volcano when GPU batch scheduling becomes contested.
- Horizontal sharding by `site_id`, `camera_id`, `event_date`, `pred_class`, or `model_version`.

### Human review

Label Studio should be treated as a stateful application:

- Run Label Studio as a `Deployment`.
- Use managed PostgreSQL where possible; otherwise run a highly backed-up PostgreSQL instance.
- Store images in object storage and pass stable URLs or signed URLs to Label Studio.
- Keep review exports versioned by project, model version, date range, and export timestamp.

### Dataset build and retraining handoff

After review, run dataset builder jobs that:

- Convert review exports to YOLO/COCO hard-negative datasets.
- Validate label schema and bbox quality.
- Write dataset manifests with source query, reviewer export, and code version.
- Publish W&B Artifacts such as `hard-negative-smoke:v12`.
- Trigger downstream training only after validation passes.

## Data Flow

```text
Inference service
  -> Event DB / Kafka
  -> Candidate query
  -> Frame/crop object store
  -> Embedding shards on GPU nodes
  -> Embedding artifact
  -> Clustering result table
  -> Label Studio tasks
  -> Review export
  -> Dataset builder
  -> W&B Artifact + model training handoff
```

Production data contracts:

- **Event metadata**: `event_id`, `camera_id`, `site_id`, `timestamp`, `bbox`, `pred_class`, `pred_confidence`, `model_version`, `frame_uri`, `crop_uri`.
- **Object URI**: stable `s3://`, `gs://`, `az://`, or HTTPS URL; avoid host-local paths in production.
- **Embedding artifact**: versioned `.npy`, parquet, or vector-store index keyed by `event_id`.
- **Review export**: immutable JSON export from Label Studio with project ID and export timestamp.
- **Dataset manifest**: machine-readable lineage file that maps training samples back to source events and review decisions.

## Storage Strategy

### On-prem first

Use on-prem object storage when frames cannot leave the data center:

- MinIO, Ceph RGW, or a commercial S3-compatible appliance.
- CSI-backed PVCs only for temporary scratch space.
- Lifecycle policies for raw crops, embeddings, and derived datasets.
- Cross-site replication only for approved or redacted artifacts.

### Cloud integrated

Use cloud storage when policy allows central dataset governance:

- Replicate curated hard negatives and metadata, not necessarily full raw video.
- Store W&B Artifacts in cloud-backed buckets.
- Keep object keys consistent across environments:

```text
s3://cv-fp-mining/raw/site=site-01/date=2026-06-16/event_id=...
s3://cv-fp-mining/embeddings/model=clip-vit-b32/date=...
s3://cv-fp-mining/reviews/labelstudio/project=fire-fp/export=...
s3://cv-fp-mining/datasets/yolo/hard-negative-smoke/version=...
```

## GPU Scheduling

Separate CPU and GPU work:

- Use a GPU node pool for embedding extraction and model-assisted mining.
- Use CPU nodes for UMAP, HDBSCAN, Label Studio exports, dataset conversion, and W&B logging.
- Add node labels such as `accelerator=nvidia`, `workload=cv-mining`, and `data-zone=onprem`.
- Use taints/tolerations so general workloads do not consume GPU nodes.
- Request GPUs explicitly with `nvidia.com/gpu`.
- Pin large model caches to node-local SSD or a read-only shared cache when possible.

Example scheduling intent:

```yaml
resources:
  limits:
    nvidia.com/gpu: "1"
nodeSelector:
  workload: cv-mining
  accelerator: nvidia
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

## Network and Security

Security posture should assume raw frames are sensitive.

- Keep raw frames on-prem unless explicitly approved for cloud replication.
- Use private connectivity between on-prem and cloud: VPN, Direct Connect, ExpressRoute, or Interconnect.
- Prefer private registry endpoints and private object-storage endpoints.
- Use mTLS or service mesh policy for internal service-to-service traffic when the cluster is shared.
- Use Kubernetes `NetworkPolicy` to restrict Label Studio, MinIO, PostgreSQL, and job namespaces.
- Use external secret managers rather than committing credentials into manifests.
- Give mining jobs short-lived credentials scoped to a date/site/class prefix.
- Encrypt object storage at rest and TLS in transit.
- Keep audit logs for object reads, review exports, dataset publication, and model promotion.

## Reliability and Operations

Production SLOs should focus on data freshness and review throughput:

- Time from production false positive to review task.
- Time from review export to curated dataset artifact.
- Number of unreviewed samples by site/class/model version.
- GPU job queue time and embedding throughput.
- Failed job retries by stage.
- Dataset validation failure rate.

Recommended operational controls:

- Idempotent jobs keyed by batch ID.
- Retry transient object-store, database, and W&B failures.
- Write checkpoint artifacts after expensive stages such as embeddings.
- Use immutable image tags or digests for mining jobs.
- Store config snapshots with each run.
- Run canary mining batches before broad schedule changes.
- Add alerts for stale schedules, failed exports, object-store replication lag, and review backlog.

## GitOps and CI/CD

Use a promotion flow that keeps code, image, and config aligned:

```text
main branch
  -> build mining image
  -> unit tests + smoke demo
  -> push image digest
  -> update staging GitOps manifest
  -> run staging mining batch
  -> promote digest to production manifest
```

Recommended layout for future infrastructure code:

```text
infra/
  helm/
    cv-fp-mining/
    label-studio/
  k8s/
    base/
    overlays/
      onprem-dev/
      onprem-prod/
      cloud-staging/
      cloud-prod/
  terraform/
    cloud-object-storage/
    container-registry/
    networking/
```

## Environment Patterns

### On-prem only

Use this when raw data cannot leave the data center.

- Kubernetes on bare metal or private virtualization.
- MinIO/Ceph for object storage.
- Label Studio and PostgreSQL inside the cluster.
- W&B offline sync or on-prem W&B.
- Cloud receives only aggregate metrics or approved dataset artifacts.

### Hybrid on-prem mining, cloud governance

Use this as the default target for most production teams.

- On-prem cluster handles frame access, embeddings, and review.
- Curated datasets, run metadata, and model lineage sync to cloud.
- Cloud CI/CD builds the mining image and deploys through GitOps.
- Cloud training consumes approved artifacts.

### Cloud burst reprocessing

Use this when historical data is already approved for cloud use.

- Copy selected frame/crop partitions to cloud object storage.
- Run the same mining image on cloud Kubernetes or batch services.
- Push review tasks to cloud Label Studio or back to on-prem review.
- Publish curated datasets through the same W&B artifact registry.

## Failure Modes and Mitigations

| Failure mode | Mitigation |
| --- | --- |
| Object paths are local-only | Use URI-based contracts and signed URLs for review |
| GPU jobs starve other workloads | Dedicated GPU node pool, quotas, Kueue/Volcano |
| Review UI loses state | External PostgreSQL, frequent backups, export snapshots |
| Large raw data leaks to cloud | Policy gates, object prefix allowlists, audit logs |
| Embedding jobs rerun too often | Cache embeddings by `event_id`, model name, and image checksum |
| Clustering is unstable between runs | Store seed, config, embedding version, and code image digest |
| CI demo is too slow | Add a tiny sample config for CI and keep full demo as nightly |
| W&B/network unavailable | Use offline mode and sync later from durable run directories |

## Migration From Current Repo

1. **Current lab**: `Justfile`, `uv`, Docker Compose, local `data/`.
2. **GPU server**: run Compose on the mining host with mounted storage and optional MinIO.
3. **On-prem Kubernetes**: package the miner as a `Job`/`CronJob`; run Label Studio and object storage as cluster services.
4. **Workflow orchestration**: split the pipeline into workflow steps with artifact handoff.
5. **Hybrid governance**: replicate curated artifacts and metadata to cloud W&B/object storage.
6. **Elastic processing**: shard embedding extraction across GPU nodes and cloud burst capacity when policy allows.

## Minimum Production Checklist

- Immutable container image and pinned `uv.lock`.
- Kubernetes manifests or Helm chart for miner jobs.
- Separate CPU and GPU node pools.
- Object storage with lifecycle and backup policy.
- External PostgreSQL for Label Studio.
- Secrets managed by Vault or cloud secret manager.
- NetworkPolicy for data, review, and job namespaces.
- W&B artifact lineage for every curated dataset.
- CI smoke tests and scheduled production canary batch.
- Runbooks for failed mining jobs, Label Studio recovery, and W&B offline sync.
