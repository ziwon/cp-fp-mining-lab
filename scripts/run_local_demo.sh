#!/usr/bin/env bash
set -euo pipefail
uv run python scripts/00_generate_sample_data.py
uv run python scripts/01_extract_embeddings.py --method simple
uv run python scripts/02_cluster_false_positives.py
uv run python scripts/03_export_for_label_studio.py
uv run python scripts/04_import_label_studio_export.py --input data/labelstudio_exports/sample_review_export.json
