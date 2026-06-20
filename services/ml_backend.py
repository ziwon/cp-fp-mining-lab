"""Label Studio ML backend (Phase 2).

Serves the current production detector so Label Studio shows live pre-labels and
uncertainty. Implements the minimal ML-backend HTTP contract (``/health``,
``/setup``, ``/predict``) without the heavy SDK.

Run::

    uv sync --extra serve
    uv run python services/ml_backend.py

Then point a Label Studio project's ML backend at http://<host>:9090.
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from flask import Flask, jsonify, request

from cv_fp_lab.config import load_config
from cv_fp_lab.registry import LocalModelRegistry
from cv_fp_lab.serving import predict_tasks


def create_app() -> Flask:
    cfg = load_config()
    registry = LocalModelRegistry(cfg["detector"]["registry_dir"])
    alias = cfg["serving"]["model_alias"]
    app = Flask(__name__)

    def _load():
        return registry.load(alias)

    @app.get("/health")
    def health():
        version = registry.stage_version(alias) or registry.production_version()
        return jsonify({"status": "UP", "model_version": version})

    @app.post("/setup")
    def setup():
        version = registry.stage_version(alias) or registry.production_version()
        return jsonify({"model_version": version})

    @app.post("/predict")
    def predict():
        payload = request.get_json(force=True) or {}
        tasks = payload.get("tasks", [])
        detector = _load()
        results = predict_tasks(tasks, detector)
        return jsonify({"results": results})

    return app


def main() -> None:
    cfg = load_config()["serving"]
    create_app().run(host=cfg["host"], port=int(cfg["port"]))


if __name__ == "__main__":
    main()
