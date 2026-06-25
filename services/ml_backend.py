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
from cv_fp_lab.serving import model_status, predict_tasks


def create_app() -> Flask:
    cfg = load_config()
    registry = LocalModelRegistry(cfg["detector"]["registry_dir"])
    alias = cfg["serving"]["model_alias"]
    app = Flask(__name__)

    def _load():
        return registry.load(alias)

    @app.get("/health")
    def health():
        status = model_status(registry, alias)
        return jsonify(status), 200 if status["ready"] else 503

    @app.post("/setup")
    def setup():
        status = model_status(registry, alias)
        return jsonify(status), 200 if status["ready"] else 503

    @app.post("/predict")
    def predict():
        status = model_status(registry, alias)
        if not status["ready"]:
            return jsonify(status), 503
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
