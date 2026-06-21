set dotenv-load

default:
    just --list

setup:
    uv sync

demo:
    uv run python scripts/00_generate_sample_data.py
    uv run python scripts/01_extract_embeddings.py --method simple
    uv run python scripts/02_cluster_false_positives.py
    uv run python scripts/03_export_for_label_studio.py
    uv run python scripts/04_import_label_studio_export.py --input data/labelstudio_exports/sample_review_export.json
    WANDB_MODE=offline uv run python scripts/05_log_to_wandb.py
    uv run python scripts/06_train_detector.py

check:
    uv run ruff check src scripts services tests
    uv run pytest

train:
    uv run python scripts/06_train_detector.py

# Real-data track (needs `uv sync --extra detect`): train a YOLO fire/smoke
# detector on D-Fire, then mine real false positives into the pipeline.
detect-prepare:
    uv run python scripts/10_prepare_dfire.py

detect-train:
    uv run python scripts/11_train_yolo_detector.py

mine-fp:
    uv run python scripts/12_mine_false_positives.py

serve-ml:
    uv run python services/ml_backend.py

serve-webhook:
    uv run python services/webhook.py

clean:
    rm -rf data/raw/*.png data/processed/* wandb .wandb

docker-build:
    docker compose build miner

docker-demo:
    docker compose run --rm miner bash -lc '\
        python scripts/00_generate_sample_data.py && \
        python scripts/01_extract_embeddings.py --method simple && \
        python scripts/02_cluster_false_positives.py && \
        python scripts/03_export_for_label_studio.py && \
        python scripts/04_import_label_studio_export.py --input data/labelstudio_exports/sample_review_export.json && \
        WANDB_MODE=offline python scripts/05_log_to_wandb.py && \
        python scripts/06_train_detector.py'

docker-shell:
    docker compose run --rm miner bash

docker-up:
    docker compose up -d minio labelstudio wandb

docker-up-all:
    docker compose up -d minio labelstudio wandb ml-backend webhook

docker-down:
    docker compose down
