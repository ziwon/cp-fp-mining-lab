"""Retrain-trigger webhook receiver (Phase 3).

Receives Label Studio annotation webhooks, debounces them into batches, and once
a batch threshold is reached runs retrain -> hold-out eval -> gated promotion via
the local model registry. The promoted model is what the ML backend serves next.

Run::

    uv sync --extra serve
    uv run python services/webhook.py

Configure a Label Studio webhook (ANNOTATION_CREATED) pointing at
http://<host>:9091/webhook.
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from flask import Flask, jsonify, request

from cv_fp_lab.config import load_config
from cv_fp_lab.feedback import ReviewBatcher, parse_annotation_event
from cv_fp_lab.registry import LocalModelRegistry
from cv_fp_lab.training import retrain_and_register


def create_app() -> Flask:
    cfg = load_config()
    processed_dir = Path(cfg["paths"]["processed_dir"])
    dcfg = cfg["detector"]
    registry = LocalModelRegistry(dcfg["registry_dir"])
    batcher = ReviewBatcher(threshold=int(cfg["webhook"]["batch_threshold"]))
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "UP", "pending": batcher.pending})

    @app.post("/webhook")
    def webhook():
        payload = request.get_json(force=True) or {}
        parsed = parse_annotation_event(payload)
        if parsed is None:
            return jsonify({"ignored": True, "pending": batcher.pending})

        batcher.add(parsed["event_id"])
        response = {"queued": parsed["event_id"], "pending": batcher.pending, "retrained": False}

        if batcher.ready():
            batch = batcher.drain()
            result = retrain_and_register(
                registry,
                events_csv=Path(cfg["embedding"]["metadata_file"]),
                embeddings_npy=Path(cfg["embedding"]["output_file"]),
                reviewed_csv=processed_dir / "reviewed_fp_samples.csv",
                metric_key=dcfg["metric_key"],
                min_delta=dcfg["promotion_min_delta"],
            )
            response.update(retrained=True, batch_size=len(batch), result=result)

        response["pending"] = batcher.pending
        return jsonify(response)

    return app


def main() -> None:
    cfg = load_config()["webhook"]
    create_app().run(host=cfg["host"], port=int(cfg["port"]))


if __name__ == "__main__":
    main()
