# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A lightweight lab demonstrating a computer-vision false-positive mining loop: collect production FP events → extract embeddings → UMAP/HDBSCAN cluster → human review via Label Studio → build a curated dataset → log lineage to W&B. It runs end-to-end on a **synthetic** dataset, so no real footage, GPU, or external accounts are required for the default path.

## Commands

Dependencies are managed with `uv`. Tasks are wrapped in a `Justfile`.

```bash
just setup        # uv sync — creates .venv/ from uv.lock
just check        # uv run ruff check src scripts tests && uv run pytest
just demo         # run the full 6-step pipeline (scripts 00→05) in offline mode
just clean        # remove generated data/raw/*.png, data/processed/*, wandb/

uv run pytest tests/test_clustering.py::test_cluster_embeddings_handles_empty_embeddings  # single test
uv run ruff check src scripts tests        # lint only (line-length 100)
```

Docker (for a GPU mining server; `miner` service reserves all NVIDIA GPUs):

```bash
just docker-build   # build the miner image
just docker-up      # start minio + labelstudio (Label Studio at http://localhost:8080)
just docker-demo    # run the full pipeline inside the container
just docker-shell   # interactive shell in the miner container
```

CI (`.github/workflows/smoke-test.yml`) runs `uv sync --frozen`, ruff, pytest, then all six pipeline scripts on every push/PR. Keep the demo runnable offline — breaking it breaks CI.

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
```

Library modules (`src/cv_fp_lab/`):
- `config.py` — loads `configs/pipeline.yaml`; scripts read all tunables from there (paths, sample counts, UMAP/HDBSCAN params, W&B project).
- `dataset.py` — synthesizes labeled FP images. `FP_COLORS`/`PRED_CLASS_BY_FP` map synthetic FP types (steam, fog, reflection, headlight, shadow, animal) to detector classes (smoke, fire, falldown, intrusion).
- `embeddings.py` — `extract_embeddings(paths, method)`. `"simple"` (default) is a deterministic offline descriptor; `"clip"` is a deliberate `NotImplementedError` extension point (needs the `clip` optional extra: `uv sync --extra clip`).
- `clustering.py` — `reduce_umap` and `cluster_embeddings` both **degrade gracefully**: UMAP falls back to SVD, HDBSCAN falls back to KMeans, and both handle empty/single-element inputs. Preserve these fallbacks — they are why the demo runs anywhere and are directly tested.
- `labelstudio.py` — `LABEL_CONFIG` (the review UI XML), `dataframe_to_labelstudio_tasks` (export; passes through optional `uncertainty`/`acquisition_rank`), `parse_labelstudio_export` (import reviewer annotations into a DataFrame).
- `active_learning.py` — `uncertainty_scores` (least-confidence/margin/entropy over `pred_confidence` as P(positive)), `rank_with_diversity` (round-robin across clusters; `cluster_id == -1` noise points compete as singletons), `select_for_review`. Drives the `--budget/--strategy/--no-diversity` flags on script 03; config under `active_learning:`. This is the implemented selection half of active learning — ML backend + webhook retraining remain future work (see `active_learning.md`).
- `wandb_logging.py` — logs the reviewed (or clustered) CSV as a Table and bundles `data/processed/` into a dataset Artifact.

## Conventions

- The package lives under `src/` (`tool.setuptools.packages.find` + pytest `pythonpath = ["src"]`); scripts add `src` to `sys.path` manually.
- Default to offline/CPU. W&B defaults to `WANDB_MODE=offline`; override with `WANDB_MODE=online WANDB_PROJECT=...`. Optional heavy deps (torch/transformers) live behind the `clip` extra.
- Add new tunables to `configs/pipeline.yaml` rather than hardcoding in scripts.
- New library code should keep the same graceful-fallback style so the offline demo and CI never require GPU or network.

Background and design notes live in `docs/` (`architecture.md`, `fp_taxonomy.md`, `hybrid_k8s_architecture.md`, `label_studio_setup.md`, `active_learning.md`). Note: the current loop is model-assisted curation (pre-annotations + clustering), **not** closed active learning; `active_learning.md` describes how to close it (Label Studio ML backend + uncertainty ranking + webhook-triggered retraining).
