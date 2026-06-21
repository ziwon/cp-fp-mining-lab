# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A lightweight lab demonstrating a computer-vision false-positive mining loop: collect production FP events → extract embeddings → UMAP/HDBSCAN cluster → human review via Label Studio → build a curated dataset → log lineage to W&B. It runs end-to-end on a **synthetic** dataset, so no real footage, GPU, or external accounts are required for the default path.

## Commands

Dependencies are managed with `uv`. Tasks are wrapped in a `Justfile`.

```bash
just setup        # uv sync — creates .venv/ from uv.lock
just check        # uv run ruff check src scripts services tests && uv run pytest
just demo         # run the full pipeline (scripts 00→06) in offline mode
just train        # bootstrap/retrain the detector and gate-promote it
just serve-ml     # Phase 2: Label Studio ML backend (needs `uv sync --extra serve`)
just serve-webhook # Phase 3: retrain-trigger webhook (needs the serve extra)
just clean        # remove generated data/raw/*.png, data/processed/*, wandb/

uv run pytest tests/test_clustering.py::test_cluster_embeddings_handles_empty_embeddings  # single test
uv run ruff check src scripts services tests   # lint only (line-length 100)
```

Docker (for a GPU mining server; `miner` service reserves all NVIDIA GPUs):

```bash
just docker-build   # build the miner image
just docker-up      # start minio + labelstudio (Label Studio at http://localhost:8080)
just docker-demo    # run the full pipeline inside the container
just docker-shell   # interactive shell in the miner container
```

CI (`.github/workflows/smoke-test.yml`) runs `uv sync --frozen --extra serve`, ruff, pytest, then all pipeline scripts (00→06) on every push/PR. Keep the demo runnable offline — breaking it breaks CI. The Flask services aren't started in CI; their logic is covered by unit tests (`test_serving.py`, `test_feedback.py`).

## Architecture

The pipeline is a chain of **numbered scripts in `scripts/`** that pass data through **files in `data/processed/`**. Each script is a thin CLI wrapper around library logic in `src/cv_fp_lab/`; put real logic in the library, keep scripts as orchestration.

Data flow (each step writes the input of the next):

```
00_generate_sample_data  → data/raw/*.png + fp_events.csv      (dataset.py)
01_extract_embeddings    → embeddings.npy                      (embeddings.py)
02_cluster_false_positives → fp_clusters.csv, fp_umap.csv      (clustering.py)
03_export_for_label_studio → labelstudio_tasks.json            (labelstudio.py)
04_import_label_studio_export → reviewed_fp_samples.csv        (labelstudio.py)
05_log_to_wandb          → W&B Table + dataset Artifact         (wandb_logging.py)
06_train_detector        → versioned model in model_registry/    (training.py, detector.py, registry.py)
```

Phases 2–3 add a live model loop **outside** the linear pipeline: `services/ml_backend.py` serves the production detector to Label Studio (predictions + uncertainty); `services/webhook.py` receives annotation webhooks, batches them, and retrains → eval → gate-promotes a new model that the backend then serves. Both are thin Flask wrappers over framework-agnostic cores in the library (`serving.py`, `feedback.py`, `training.py`) so the logic is unit-tested without standing up servers.

Library modules (`src/cv_fp_lab/`):
- `config.py` — loads `configs/pipeline.yaml`; scripts read all tunables from there (paths, sample counts, UMAP/HDBSCAN params, W&B project).
- `dataset.py` — synthesizes labeled FP images. `FP_COLORS`/`PRED_CLASS_BY_FP` map synthetic FP types (steam, fog, reflection, headlight, shadow, animal) to detector classes (smoke, fire, falldown, intrusion).
- `embeddings.py` — `extract_embeddings(paths, method)`. `"simple"` (default) is a deterministic offline descriptor; `"clip"` is a deliberate `NotImplementedError` extension point (needs the `clip` optional extra: `uv sync --extra clip`).
- `clustering.py` — `reduce_umap` and `cluster_embeddings` both **degrade gracefully**: UMAP falls back to SVD, HDBSCAN falls back to KMeans, and both handle empty/single-element inputs. Preserve these fallbacks — they are why the demo runs anywhere and are directly tested.
- `labelstudio.py` — `LABEL_CONFIG` (the review UI XML), `dataframe_to_labelstudio_tasks` (export; passes through optional `uncertainty`/`acquisition_rank`), `parse_labelstudio_export` (import reviewer annotations into a DataFrame).
- `active_learning.py` — `uncertainty_scores` (least-confidence/margin/entropy over `pred_confidence` as P(positive)), `rank_with_diversity` (round-robin across clusters; `cluster_id == -1` noise points compete as singletons), `select_for_review`. Drives the `--budget/--strategy/--no-diversity` flags on script 03; config under `active_learning:`. The selection half of active learning.
- `detector.py` — `FpDetector`: sklearn `StandardScaler + LogisticRegression` predicting `fp_type` from embeddings. Provides hold-out metrics (`train` refits on all data after evaluating), `confidence`, normalized-entropy `uncertainty`, and joblib `save`/`load`. This is the trainable/servable model for Phases 2–3.
- `yolo_detector.py` — `YoloDetector`: thin Ultralytics YOLO adapter (`train`/`load`/`predict`) + `write_dfire_yaml`. The **real** fire/smoke detector for the D-Fire track (needs `uv sync --extra detect`). Used by scripts 10–12, not by the synthetic pipeline.
- `fp_mining.py` — real false-positive harvesting: `iou_xyxy`, `read_yolo_labels`, `is_false_positive` (no GT match → FP), and `collect_false_positives` (run detector over a YOLO split, crop FPs, emit the `fp_events.csv` schema). Replaces the synthetic generator (script 00) with mined real FPs; downstream 01→05 unchanged.
- `registry.py` — `LocalModelRegistry`: filesystem model store (`<root>/models/<version>/`, `aliases.json`) with `candidate/staging/production` stages. `maybe_promote` registers a candidate and promotes only if its `metric_key` beats production by `min_delta` (offline stand-in for the W&B Registry).
- `training.py` — `assemble_training_data` (bootstrap labels from `synthetic_fp_type`, override with human `review_fp_type`) and `retrain_and_register` (train → gate-promote).
- `serving.py` — `predict_tasks` maps Label Studio tasks to ML-backend predictions (embed image → classify → LS choices + score + uncertainty meta).
- `feedback.py` — `parse_annotation_event` (LS webhook → `{event_id, review_fp_type}`) and `ReviewBatcher` (dedupe + threshold debounce for retraining).
- `wandb_logging.py` — logs the reviewed (or clustered) CSV as a Table and bundles `data/processed/` into a dataset Artifact.

## Conventions

- The package lives under `src/` (`tool.setuptools.packages.find` + pytest `pythonpath = ["src"]`); scripts add `src` to `sys.path` manually.
- Default to offline/CPU. W&B defaults to `WANDB_MODE=offline`; override with `WANDB_MODE=online WANDB_PROJECT=...`. Heavy/optional deps live behind extras: torch/transformers in `clip`, Flask (for the services) in `serve`, ultralytics (real YOLO detector) in `detect`.
- A real-data track (`scripts/10_prepare_dfire`, `11_train_yolo_detector`, `12_mine_false_positives`) trains a YOLO fire/smoke detector on D-Fire and mines its real false positives into the same `fp_events.csv` the synthetic pipeline uses. See `docs/real_data.md`. On this host heavy artifacts go to `/data` (small workspace disk).
- Add new tunables to `configs/pipeline.yaml` rather than hardcoding in scripts.
- New library code should keep the same graceful-fallback style so the offline demo and CI never require GPU or network. Keep HTTP services as thin wrappers over testable library cores.

Background and design notes live in `docs/` (`architecture.md`, `fp_taxonomy.md`, `hybrid_k8s_architecture.md`, `label_studio_setup.md`, `active_learning.md`, `demo_walkthrough.md`, `real_data.md`). The active-learning loop is implemented end-to-end offline: selection (script 03), serving (`services/ml_backend.py`), and webhook retraining (`services/webhook.py`); `active_learning.md` documents the design and the production hardening still left (W&B-backed registry, GPU serving, dataset validation, Kubernetes).
