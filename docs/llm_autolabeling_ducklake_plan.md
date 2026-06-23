# LLM Autolabeling and DuckLake Planning

This document proposes two production-oriented extensions for the false-positive
mining lab:

1. Add LLM vision autolabeling as a weak-label/pre-annotation layer.
2. Add DuckLake as the queryable metadata and lineage layer for mined events,
   labels, review batches, datasets, and gate results.

The intent is not to remove human review. The safer shape is to use LLM labels to
triage and accelerate review, while DuckLake records every intermediate decision
so training data and promotion decisions are reproducible.

## Current repo fit

The repo already has the main integration points:

- `src/cv_fp_lab/labelstudio.py` writes Label Studio tasks with a `predictions`
  block. This is the natural place to include LLM-generated pre-annotations.
- `src/cv_fp_lab/serving.py` maps Label Studio tasks to ML-backend predictions.
  The current `FpDetector` path can be extended with an LLM vision predictor or
  an ensemble of detector + LLM predictions.
- `src/cv_fp_lab/active_learning.py` ranks review candidates by uncertainty and
  cluster diversity. LLM confidence and detector/LLM disagreement can become
  additional acquisition signals.
- `configs/pipeline.yaml` already centralizes tunables. LLM provider/model,
  thresholds, prompt version, DuckLake catalog path, and data path should live
  there rather than in scripts.
- W&B should remain the model/artifact tracking surface. DuckLake should sit
  before W&B as the local/queryable lakehouse of candidate events, labels,
  review state, and dataset snapshots.

## Option 1: LLM vision autolabeling for Label Studio pre-annotations

### Goal

Use a multimodal LLM to produce structured pre-annotations for FP crops:

- `is_event`: `real_event`, `false_positive`, or `uncertain`
- `fp_type`: `steam`, `fog`, `dust`, `reflection`, `headlight`, `shadow`,
  `sitting`, `animal`, `authorized_worker`, or `unknown`
- `bbox_valid`: `valid`, `wrong_class`, `wrong_location`, or `unnecessary`
- `confidence`: calibrated enough for routing, not treated as ground truth
- `needs_human_review`: always true at first, then thresholded after validation
- `comment`: short reviewer-facing explanation

The output should be imported into Label Studio as predictions, not annotations,
so humans can accept, correct, or override it.

### Recommended design

Add a small library module and one script:

```text
src/cv_fp_lab/llm_autolabeling.py
scripts/07_autolabel_with_llm.py
```

The script should read `data/processed/fp_clusters.csv` or
`data/processed/labelstudio_tasks.json`, call the LLM vision model on each crop,
validate the response against a strict schema, and write:

```text
data/processed/llm_autolabels.csv
data/processed/labelstudio_tasks.llm.json
```

The Label Studio task export should keep the existing detector fields and add LLM
metadata under `data`:

```json
{
  "llm_model": "vision-model-name",
  "llm_prompt_version": "fp-review-v1",
  "llm_confidence": 0.87,
  "llm_needs_human_review": true
}
```

The `predictions` block should contain the LLM's proposed choices in the same
format that `dataframe_to_labelstudio_tasks()` already emits.

### Routing policy

Start with conservative routing:

- High-confidence agreement: detector/cluster prior and LLM agree, and
  `llm_confidence >= 0.90`. Send to a fast-review queue.
- Disagreement: detector and LLM disagree, or LLM says `uncertain`. Prioritize for
  human review.
- Low-confidence: `llm_confidence < 0.70`. Prioritize for human review.
- Unknown/new pattern: LLM chooses `unknown` or gives low confidence inside a
  dense cluster. Prioritize for taxonomy updates.

This can extend the active-learning score:

```text
review_score =
  alpha * detector_uncertainty
  + beta * cluster_diversity_score
  + gamma * detector_llm_disagreement
  + delta * llm_low_confidence
```

The existing `select_for_review()` API can remain the main selection path; add
optional columns such as `llm_confidence`, `llm_label`, and `llm_disagreement`.

### Prompt and schema guidance

The prompt should be short and constrained:

- Describe the operational context: fire/smoke/safety detector false-positive
  review.
- Provide the allowed enums and their definitions.
- Ask the model to classify only visible evidence in the crop.
- Require `uncertain` when the crop is insufficient.
- Avoid long reasoning in stored outputs; keep a short `comment`.

Use structured outputs so invalid enum values and missing fields fail fast. Store
the raw model response for audit, but only feed validated normalized fields into
Label Studio and training data.

### Evaluation

Before using LLM labels for any automation, run a validation pass on already
reviewed Label Studio exports:

- Agreement with human `review_is_event`
- Agreement with human `review_fp_type`
- Agreement with human `review_bbox_valid`
- Per-class precision/recall for high-confidence LLM predictions
- Error rate by `pred_class`, `camera_id`, `site_id`, and cluster

Promotion from "review assist" to "auto-accept for low-risk cases" should require
stable high precision on held-out human-reviewed data. Even then, keep human audit
sampling.

### Risks

- LLMs may over-interpret ambiguous crops.
- Confidence may not be calibrated.
- Site/camera-specific artifacts can create systematic errors.
- Sending production images to an external API may require privacy review.
- Cost can spike if every crop is labeled individually.

Mitigations:

- Cache by image hash and prompt version.
- Batch only selected review candidates, not the entire event stream.
- Keep the default path offline and deterministic for CI.
- Treat LLM outputs as predictions until validated against human labels.

## Option 2: DuckLake for queryable lineage and dataset snapshots

### Goal

Move the pipeline's metadata from disconnected CSV/JSON files into a small
lakehouse that can answer questions such as:

- Which event candidates were mined by model version `X`?
- Which LLM prompt/model produced each weak label?
- Which human-reviewed labels were used to build dataset version `Y`?
- What changed between two review batches?
- Which hard-negative dataset produced the candidate that passed or failed the
  gate?

DuckLake is a good fit because it lets DuckDB attach to a lakehouse catalog and
query versioned tables using SQL. It also supports lakehouse-style features such
as time travel and querying changes between snapshots.

### Recommended design

Add an optional `lineage` extra:

```toml
[project.optional-dependencies]
lineage = [
    "duckdb>=1.3",
]
```

Add:

```text
src/cv_fp_lab/lineage.py
scripts/08_sync_to_ducklake.py
```

The local default should be file-backed and offline:

```yaml
ducklake:
  catalog: data/processed/ducklake/metadata.ducklake
  data_path: data/processed/ducklake/data
```

Production can later switch the data path to object storage and the catalog to
PostgreSQL for multi-user access.

### Proposed tables

```text
events
  event_id, camera_id, site_id, timestamp, image_uri, image_local_path,
  source_image_path, bbox, pred_class, pred_confidence, model_version,
  is_negative_image, mined_at, mining_run_id

embeddings
  event_id, embedding_model, embedding_method, embedding_uri, embedding_dim,
  embedding_run_id, created_at

clusters
  event_id, cluster_id, umap_x, umap_y, clustering_run_id, created_at

acquisition_queue
  event_id, strategy, uncertainty, acquisition_rank, budget, diversity,
  selection_run_id, created_at

llm_autolabels
  event_id, llm_provider, llm_model, prompt_version, is_event, fp_type,
  bbox_valid, confidence, needs_human_review, comment, raw_response_uri,
  autolabel_run_id, created_at

human_reviews
  event_id, annotation_id, reviewer_id, review_is_event, review_fp_type,
  review_bbox_valid, review_comment, labelstudio_project_id, reviewed_at

training_datasets
  dataset_version, source_snapshot_id, source_review_batch_id, yolo_yaml,
  artifact_uri, n_images, n_empty_labels, created_at

gate_results
  candidate_model_version, production_model_version, dataset_version,
  map50, map5095, recall_smoke, recall_fire, neg_fp_rate,
  promoted, gate_run_id, created_at
```

### Pipeline placement

The initial implementation can be append/sync based:

```text
00/12 mining             -> events
01 embeddings            -> embeddings
02 clustering            -> clusters
03 active-learning export -> acquisition_queue
07 LLM autolabeling       -> llm_autolabels
04 LS import              -> human_reviews
14 hard-negative build    -> training_datasets
13 evaluate/gate          -> gate_results
```

This keeps the current file-based pipeline intact. DuckLake becomes a durable
index over the files rather than a hard runtime dependency.

### Example queries

Find high-confidence LLM labels that humans corrected:

```sql
SELECT
  l.event_id,
  l.fp_type AS llm_fp_type,
  h.review_fp_type,
  l.confidence,
  e.pred_class,
  c.cluster_id
FROM llm_autolabels l
JOIN human_reviews h USING (event_id)
JOIN events e USING (event_id)
LEFT JOIN clusters c USING (event_id)
WHERE l.confidence >= 0.9
  AND l.fp_type <> h.review_fp_type;
```

Find clusters with many reviewed false positives but no hard-negative dataset
coverage:

```sql
SELECT
  c.cluster_id,
  COUNT(*) AS reviewed_fp_count
FROM clusters c
JOIN human_reviews h USING (event_id)
WHERE h.review_is_event = 'false_positive'
GROUP BY c.cluster_id
ORDER BY reviewed_fp_count DESC;
```

Audit the source of a promoted model:

```sql
SELECT
  g.candidate_model_version,
  g.promoted,
  g.map50,
  g.neg_fp_rate,
  d.dataset_version,
  d.source_snapshot_id,
  d.artifact_uri
FROM gate_results g
JOIN training_datasets d USING (dataset_version)
WHERE g.promoted = true
ORDER BY g.created_at DESC;
```

### Why DuckLake instead of only W&B

W&B remains the right surface for artifacts, runs, charts, and model registry
handoff. DuckLake is better for local SQL questions over evolving row-level
metadata:

- event-level joins across mining, clustering, LLM labels, human review, and gate
  results
- reproducible dataset snapshots before they become W&B artifacts
- querying data changes between review batches
- retaining local/offline operation for the demo and CI
- later migration to object storage and PostgreSQL-backed metadata

## Implementation phases

### Phase 1: LLM labels as static pre-annotations

- Add `llm_autolabeling.py` with schema validation, image loading, caching, and
  provider abstraction.
- Add `scripts/07_autolabel_with_llm.py`.
- Add config fields for provider, model, prompt version, confidence thresholds,
  max images, and cache directory.
- Extend Label Studio task export to include LLM predictions and metadata.
- Add tests with a fake LLM client so CI stays offline.

### Phase 2: LLM-aware active learning

- Add detector/LLM disagreement and LLM confidence to review ranking.
- Write `llm_autolabels.csv` and `labelstudio_ranking.csv` together.
- Evaluate LLM/human agreement on reviewed exports.
- Add a report script for per-class and per-cluster LLM error analysis.

### Phase 3: DuckLake metadata sync

- Add `lineage.py` and `scripts/08_sync_to_ducklake.py`.
- Create local DuckLake catalog under `data/processed/ducklake/`.
- Sync existing CSV/JSON outputs into normalized tables.
- Add smoke tests for empty inputs and schema evolution.

### Phase 4: Snapshot-based dataset building

- Record the DuckLake snapshot used to build each hard-negative dataset.
- Store the snapshot ID in W&B artifact metadata.
- Make gate reports link candidate model -> dataset version -> DuckLake snapshot
  -> source events and human reviews.

## References

- Label Studio pre-annotations: https://labelstud.io/guide/predictions
- Label Studio ML backend: https://labelstud.io/guide/ml_create
- Label Studio ML pipeline integration: https://labelstud.io/guide/ml
- OpenAI structured outputs: https://developers.openai.com/api/docs/guides/structured-outputs
- OpenAI vision inputs: https://developers.openai.com/api/docs/guides/images-vision
- DuckLake DuckDB extension: https://duckdb.org/docs/current/core_extensions/ducklake.html
- DuckLake introduction: https://ducklake.select/docs/stable/duckdb/introduction.html
- DuckLake connecting and `ATTACH`: https://ducklake.select/docs/stable/duckdb/usage/connecting.html
- DuckLake time travel: https://ducklake.select/docs/stable/duckdb/usage/time_travel.html
- DuckLake data change feed: https://ducklake.select/docs/stable/duckdb/advanced_features/data_change_feed.html
