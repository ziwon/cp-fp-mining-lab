# Production Architecture with Argo Workflows and DuckLake

This document turns the current `cv-fp-mining-lab` architecture into a
production platform plan. It preserves the lab's important property: the default
path stays offline, deterministic, and runnable from numbered scripts. The
production path adds orchestration, durable lineage, review operations, and
promotion controls around the same mining loop.

![Production architecture overview](assets/architecture-overview.svg)

## Current Repo Fit

The current repository already contains most of the platform's domain boundary
lines:

| Production capability | Current implementation | Production gap |
| --- | --- | --- |
| Event mining | `scripts/00_generate_sample_data.py`, `scripts/12_mine_false_positives.py`, `fp_mining.py` | Replace CSV/local paths with idempotent event ingestion and URI-based object references |
| Embeddings | `embeddings.py`, `scripts/01_extract_embeddings.py` | Shard GPU extraction by site/date/model version and cache by image checksum |
| Clustering | `clustering.py`, `scripts/02_cluster_false_positives.py` | Persist cluster run metadata, representative samples, and recurrence statistics |
| Active learning | `active_learning.py`, `scripts/03_export_for_label_studio.py` | Add recurrence, site rarity, review backlog, Cleanlab, and optional LLM/VLM disagreement signals |
| Human review | `labelstudio.py`, Label Studio Docker service | Add review sync worker, webhook import, schema validation, and idempotent merges |
| Dataset build | `dataset_builder.py`, `scripts/14_build_hard_negatives.py` | Produce immutable manifests bound to DuckLake snapshot IDs |
| Training and gating | `training.py`, `gating.py`, scripts `06`, `13`, `15` | Move to Argo workflows with protected eval sets, approval metadata, and registry promotion records |
| Experiment tracking | `wandb_logging.py`, local registry | Keep W&B or MLflow for runs/artifacts/models, but do not make it the dataset source of truth |

The main architecture change is not a rewrite of the mining logic. It is the
addition of durable contracts around it: object URIs, lakehouse tables, workflow
templates, review synchronization, and explicit promotion gates.

## Improved Target Architecture

The production platform is split into layers:

1. **External production layer**: CCTV/VMS, edge devices, production inference,
   event source, and model runtime.
2. **Access and API layer**: event ingest API, webhook receiver, admin/config
   API, taxonomy and label policy API.
3. **Workflow orchestration layer**: Argo Workflows, WorkflowTemplates,
   CronWorkflows, event-triggered workflows.
4. **Mining and active-learning workflows**: candidate mining, frame/crop
   resolution, embedding extraction, clustering, acquisition ranking, and
   quality auditing.
5. **Review and annotation layer**: Label Studio, ML backend, optional LLM/VLM
   pre-labeler, review sync worker.
6. **Dataset/training/evaluation workflows**: immutable dataset build, dataset
   validation, training, evaluation gate, model promotion.
7. **Lakehouse and metadata layer**: DuckLake tables, PostgreSQL catalog and
   application metadata, stateless DuckDB runtime inside workflow pods.
8. **Object storage layer**: MinIO/S3-compatible storage for frames, crops,
   embeddings, Parquet, datasets, and model artifacts.
9. **ML lifecycle layer**: W&B or MLflow for runs, visual reports, artifacts,
   and registry stages.
10. **Operations/security layer**: observability, RBAC, secrets, backup,
    retention, audit, and data quality maintenance.

## Production Workflow

![Production workflow](assets/production-workflow.svg)

The batch path is:

```text
production events
  -> candidate mining
  -> frame/crop resolve
  -> embedding shards
  -> UMAP/HDBSCAN or fallback clustering
  -> active-learning acquisition queue
  -> Label Studio review
  -> review sync into DuckLake
  -> dataset manifest build
  -> training
  -> evaluation gate
  -> model registry promotion
```

Argo should own the production DAG. Individual workflow steps can continue to
call the existing Python scripts while the library modules remain the testable
source of business logic.

Recommended workflow families:

| Workflow | Trigger | Notes |
| --- | --- | --- |
| Candidate mining | hourly/nightly CronWorkflow or event trigger | Query event metadata by site/date/class/model version |
| Frame and crop resolve | mining batch dependency | Normalize frame/crop objects and assign stable URIs |
| Embedding extraction | batch dependency, GPU shards | Cache by `event_id`, image checksum, embedding model, and version |
| Clustering and ranking | embedding completion | Combine uncertainty, diversity, recurrence, and quality signals |
| Review export | ranking completion | Create Label Studio tasks with detector and optional LLM/VLM predictions |
| Review sync | Label Studio webhook or scheduled import | Validate and merge annotations idempotently |
| Dataset build | review batch ready | Emit YOLO/COCO data plus machine-readable manifest |
| Evaluation gate | training completion | Require mAP no-regression, recall floors, FP-rate non-increase, and latency checks when relevant |
| Maintenance | scheduled | DuckLake compaction, snapshot retention, object cleanup, backup tests, freshness checks |

## Deployment Progression

![Deployment topology](assets/deployment-topology.svg)

The safest adoption path is incremental:

1. **Local lab**: keep synthetic data, local files, Docker Compose, Label Studio,
   and W&B offline/online logging.
2. **Single GPU server**: run the existing miner image with host-mounted storage
   or MinIO; use GPU embedding extraction where available.
3. **Shared object storage**: replace host-local paths with `s3://`, `gs://`,
   `az://`, or signed HTTPS URLs; keep workers stateless.
4. **Kubernetes batch execution**: run the numbered pipeline as Argo workflow
   steps; split CPU and GPU stages; add retries and observability.
5. **DuckLake lineage**: sync events, embeddings, clusters, acquisition queues,
   reviews, datasets, and gate results into DuckLake tables; bind dataset
   manifests to snapshot IDs.
6. **Production ML lifecycle**: formalize training, protected evaluation,
   promotion approvals, and W&B/MLflow registry stages.
7. **Full platform**: add HA PostgreSQL, PITR backups, NetworkPolicy, RBAC,
   retention policies, cross-site replication where approved, and operational
   dashboards.

## Lakehouse Contract

DuckLake is the dataset and row-level lineage source of truth. W&B or MLflow is
the experiment, artifact, report, and registry source of truth. The tracker
should log references to DuckLake dataset versions and snapshots rather than
becoming the canonical event database.

![DuckLake lineage model](assets/lakehouse-lineage.svg)

Recommended logical tables:

| Table | Purpose | Key fields |
| --- | --- | --- |
| `events` | Production event metadata and object references | `event_id`, `camera_id`, `site_id`, `timestamp`, `frame_uri`, `crop_uri`, `source_system`, `ingested_at` |
| `detector_predictions` | Model prediction outputs | `event_id`, `model_version`, `pred_class`, `pred_confidence`, `bbox`, `prediction_timestamp` |
| `embeddings` | Embedding metadata and vector object references | `event_id`, `embedding_model`, `embedding_version`, `embedding_uri`, `vector_dim`, `created_at` |
| `clusters` | Cluster assignments and representative scores | `event_id`, `cluster_id`, `cluster_method`, `cluster_version`, `distance_to_centroid`, `representative_score` |
| `acquisition_queue` | Review candidate ranking | `event_id`, `review_batch_id`, `rank_score`, `ranking_reason`, `priority`, `status` |
| `llm_prelabels` | Optional LLM/VLM review assistance | `event_id`, `oracle_model`, `prompt_version`, `pre_label`, `confidence`, `rationale`, `created_at` |
| `human_reviews` | Human-reviewed labels and comments | `event_id`, `annotation_id`, `review_batch_id`, `reviewer_id`, `label`, `fp_type`, `bbox_valid`, `comment`, `reviewed_at` |
| `dataset_manifests` | Immutable training/eval dataset versions | `dataset_version`, `snapshot_id`, `source_query`, `review_batch_id`, `manifest_uri`, `train_split_uri`, `eval_split_uri`, `created_at` |
| `eval_gate_results` | Evaluation and promotion decisions | `model_version`, `dataset_version`, `eval_snapshot_id`, `metric_name`, `metric_value`, `threshold`, `passed`, `promotion_decision`, `created_at` |

DuckDB should remain a stateless runtime inside workflow containers. PostgreSQL
backs the DuckLake catalog and application metadata in production; local file
catalogs are enough for the lab and CI.

## Object Storage Contract

Object storage should use stable URI-based paths, not host-local paths:

```text
s3://cv-fp-mining/raw/site=<site>/date=<date>/event_id=<event_id>/frame.jpg
s3://cv-fp-mining/crops/site=<site>/date=<date>/event_id=<event_id>/crop.jpg
s3://cv-fp-mining/embeddings/model=<model>/date=<date>/part-000.parquet
s3://cv-fp-mining/datasets/type=hard-negative/class=<class>/version=<version>/manifest.json
```

Recommended prefixes:

```text
raw/             frames, clips, original event payloads
crops/           bounding-box crops for review and embedding
embeddings/      vector files, feature files, optional indexes
ducklake-data/   Parquet files managed by DuckLake
datasets/        COCO, YOLO, split manifests, hard-negative datasets
ml-artifacts/    trained models, evaluation reports, exported charts
```

Raw frames and crops may be sensitive. Keep them on-premises unless policy
explicitly permits replication. Curated metadata, aggregate metrics, and
approved dataset artifacts can move to a central environment when allowed.

## Review and Assisted Labeling

Label Studio remains the human review UI, not the source of truth for training
datasets. The durable source of truth is the reviewed-label table after the
review sync worker validates and merges Label Studio exports into DuckLake.

Recommended review fields:

```text
event_valid
fp_type
bbox_valid
correct_class
review_comment
reviewer_id
review_timestamp
```

The ML backend and optional LLM/VLM pre-labeler should produce predictions, not
authoritative annotations. Store LLM/VLM outputs separately with:

```text
oracle_type
oracle_model
prompt_version
pre_label
confidence
rationale
created_at
```

Use LLM/VLM labels conservatively at first:

- route high-confidence agreement to faster review
- prioritize detector/LLM disagreement
- prioritize low-confidence or `unknown` outputs
- keep human audit sampling before any auto-accept policy

## Active-Learning Improvements

The attached plan is directionally sound. The main improvement is to treat
acquisition ranking as a composable score rather than a cluster-only selection
step:

```text
review_score =
  detector_uncertainty
  + cluster_diversity
  + recurrence_score
  + site_or_camera_rarity
  + detector_llm_disagreement
  + label_quality_risk
  + backlog_priority
```

Keep clustering for diversity and deduplication. Use uncertainty, recurrence,
review backlog, and quality signals to decide which samples become review work.
Noise points (`cluster_id == -1`) should compete as individual candidates so
novel failure modes are not dropped.

## Evaluation and Promotion Policy

Promotion must be multi-criteria. A model should not pass simply because it
reduces false positives by detecting nothing.

Recommended gate checks:

- no mAP regression beyond tolerance
- per-class recall floors for protected classes
- negative-image FP-rate non-increase
- false-negative impact analysis
- site/camera-specific regression checks for critical deployments
- latency/resource checks when the serving target is constrained
- promotion metadata: dataset version, DuckLake snapshot, baseline model,
  candidate model, approval status, promoter, rollback target

This aligns with the existing `gating.py` direction and should remain a pure,
testable library core under any Argo wrapper.

## Kubernetes Layout

Recommended namespaces:

```text
cv-mining-system     API services, webhook receiver, workflow templates, shared config
cv-mining-jobs       Argo workflow pods for mining, embedding, clustering, ranking, datasets
cv-review            Label Studio, Label Studio DB, ML backend, review sync worker
cv-data              MinIO/S3 gateway, PostgreSQL, DuckLake catalog, metadata services
cv-ml-lifecycle      W&B or MLflow, model registry, reports
cv-observability     Prometheus, Grafana, Loki, OpenTelemetry, Alertmanager
cv-security          External Secrets, policy controllers, audit tooling
```

Separate CPU and GPU work:

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

## Operations, Security, and Maintenance

Production SLOs should focus on freshness, throughput, and safety:

- time from production false positive to review task
- review backlog by site/class/model version
- time from review completion to dataset publication
- embedding throughput and GPU queue time
- workflow failure rate and retry count by stage
- dataset validation failure rate
- evaluation gate pass/fail rate
- object storage read/write errors
- PostgreSQL replication lag
- DuckLake compaction status
- W&B/MLflow logging failures

Required controls:

- idempotent ingest and review sync keyed by `event_id`, `model_version`,
  `annotation_id`, `review_batch_id`, and `label_version`
- immutable container image tags or digests
- config snapshots stored with each workflow run
- short-lived credentials scoped to object prefixes
- encrypted object storage and TLS in transit
- Kubernetes NetworkPolicy and RBAC per namespace
- HA PostgreSQL with backup, PITR, restore tests, and monitoring
- scheduled DuckLake compaction and snapshot retention workflows
- object lifecycle policies for raw, intermediate, and curated artifacts
- audit logs for object reads, review exports, dataset publication, and model
  promotion

## Recommended Repository Expansion

A production repo can grow from this lab without moving all code at once:

```text
workflows/
  templates/
  cronworkflows/
  event-triggered/
infra/
  helm/
  kustomize/
  terraform/
configs/
  taxonomy.yaml
  acquisition.yaml
  dataset.yaml
  training.yaml
docs/
  data-contracts.md
  lakehouse-schema.md
  workflow-design.md
  security-governance.md
  operations-runbook.md
```

Do this after the local contracts are proven. The immediate next production
implementation step is an optional DuckLake sync layer that indexes the existing
CSV/JSON outputs without making the offline demo depend on new services.

## Architecture Decisions

1. **Use Argo Workflows for production orchestration.** It gives DAG execution,
   retries, artifact passing, GPU/CPU separation, CronWorkflows, and workflow
   audit trails.
2. **Use DuckLake as the dataset lineage layer.** It provides queryable tables,
   snapshots, and reproducible dataset versions over object storage.
3. **Keep DuckDB stateless.** Run DuckDB inside workflow containers rather than
   operating it as a central service.
4. **Treat Label Studio as a review system.** Export and merge labels into the
   lakehouse before building training datasets.
5. **Keep W&B or MLflow as ML lifecycle tracking.** Use it for runs, artifacts,
   reports, and model registry state; store row-level dataset truth in DuckLake.
6. **Separate raw, curated, and artifact storage.** Apply distinct retention,
   replication, and access policies to each class of data.

## Summary

The production architecture keeps the lab's file-based pipeline as the smallest
portable execution unit, then wraps it with Argo orchestration, DuckLake lineage,
object storage contracts, Label Studio sync, dataset validation, evaluation
gates, and operational controls. The result is a closed data-centric improvement
loop from production model failures to reviewed datasets, retraining, promotion,
and auditable feedback.
