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
from cv_fp_lab.training import persist_review_labels, retrain_and_register
from cv_fp_lab.wandb_logging import log_retrain_run


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

        batcher.add(parsed["event_id"], parsed["review_fp_type"])
        response = {"queued": parsed["event_id"], "pending": batcher.pending, "retrained": False}

        if batcher.ready():
            reviews = batcher.drain()
            # Persist the corrected labels so the retrain actually learns from them.
            reviews_csv = processed_dir / "webhook_reviews.csv"
            persist_review_labels(reviews_csv, reviews)
            result = retrain_and_register(
                registry,
                events_csv=Path(cfg["embedding"]["metadata_file"]),
                embeddings_npy=Path(cfg["embedding"]["output_file"]),
                reviewed_csv=reviews_csv,
                metric_key=dcfg["metric_key"],
                min_delta=dcfg["promotion_min_delta"],
            )
            # Log the retrain as a W&B run (no-op offline / if wandb unavailable).
            try:
                result["wandb_url"] = log_retrain_run(
                    project=cfg["wandb"]["project"],
                    result=result,
                    registry_dir=dcfg["registry_dir"],
                )
            except Exception as exc:  # keep the loop alive if W&B is down
                result["wandb_error"] = str(exc)
            response.update(retrained=True, batch_size=len(reviews), result=result)

        response["pending"] = batcher.pending
        return jsonify(response)

    return app


def main() -> None:
    cfg = load_config()["webhook"]
    create_app().run(host=cfg["host"], port=int(cfg["port"]))


if __name__ == "__main__":
    main()
