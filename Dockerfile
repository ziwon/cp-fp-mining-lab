FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PROJECT_ENVIRONMENT=/usr/local \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
RUN python -m pip install --upgrade pip uv \
    && uv sync --frozen --no-dev --no-install-project

COPY configs ./configs
COPY scripts ./scripts
COPY src ./src
COPY data/labelstudio_exports ./data/labelstudio_exports

CMD ["bash"]
