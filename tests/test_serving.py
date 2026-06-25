import numpy as np

from cv_fp_lab.detector import FpDetector
from cv_fp_lab.registry import LocalModelRegistry
from cv_fp_lab.serving import model_status, predict_tasks


def _detector():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, 0.1, (10, 3)), rng.normal(5, 0.1, (10, 3))])
    y = np.array(["steam"] * 10 + ["fog"] * 10)
    return FpDetector.train(x, y)


def test_predict_tasks_returns_labelstudio_predictions() -> None:
    det = _detector()
    # Embed function keyed by the task image string, bypassing real image IO.
    vectors = {"steam.png": np.zeros(3), "fog.png": np.full(3, 5.0)}
    tasks = [{"data": {"image": "steam.png"}}, {"data": {"image": "fog.png"}}]

    preds = predict_tasks(tasks, det, embed_fn=lambda p: vectors[p])

    assert len(preds) == 2
    assert preds[0]["result"][0]["value"]["choices"] == ["steam"]
    assert preds[1]["result"][0]["value"]["choices"] == ["fog"]
    assert preds[0]["model_version"] == det.model_version
    assert 0.0 <= preds[0]["meta"]["uncertainty"] <= 1.0
    assert preds[0]["result"][0]["from_name"] == "fp_type"


def test_predict_tasks_handles_empty() -> None:
    assert predict_tasks([], _detector()) == []


def test_model_status_reports_missing_and_ready_model(tmp_path) -> None:
    reg = LocalModelRegistry(tmp_path)

    missing = model_status(reg)
    assert missing["ready"] is False
    assert missing["status"] == "NOT_READY"

    reg.maybe_promote(_detector())
    ready = model_status(reg)
    assert ready["ready"] is True
    assert ready["status"] == "UP"
    assert ready["model_version"] == reg.production_version()
