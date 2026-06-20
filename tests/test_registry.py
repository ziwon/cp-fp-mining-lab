import numpy as np

from cv_fp_lab.detector import FpDetector
from cv_fp_lab.registry import LocalModelRegistry


def _detector(metric: float, seed: int = 0) -> FpDetector:
    rng = np.random.default_rng(seed)
    x = np.vstack(
        [rng.normal(0, 0.1, (10, 3)), rng.normal(5, 0.1, (10, 3))]
    )
    y = np.array(["steam"] * 10 + ["fog"] * 10)
    det = FpDetector.train(x, y)
    det.metrics["macro_f1"] = metric  # override for deterministic gate tests
    return det


def test_first_model_is_promoted(tmp_path) -> None:
    reg = LocalModelRegistry(tmp_path)
    res = reg.maybe_promote(_detector(0.5))
    assert res["promoted"] is True
    assert res["reason"] == "first model"
    assert reg.production_version() == res["version"]


def test_better_candidate_promotes_worse_does_not(tmp_path) -> None:
    reg = LocalModelRegistry(tmp_path)
    first = reg.maybe_promote(_detector(0.70, seed=1))
    prod_after_first = reg.production_version()

    worse = reg.maybe_promote(_detector(0.60, seed=2))
    assert worse["promoted"] is False
    assert reg.production_version() == prod_after_first  # unchanged

    better = reg.maybe_promote(_detector(0.85, seed=3))
    assert better["promoted"] is True
    assert reg.production_version() == better["version"]
    assert better["version"] != first["version"]


def test_min_delta_gate(tmp_path) -> None:
    reg = LocalModelRegistry(tmp_path)
    reg.maybe_promote(_detector(0.80, seed=1))
    # Equal metric does not clear a positive min_delta.
    res = reg.maybe_promote(_detector(0.80, seed=2), min_delta=0.05)
    assert res["promoted"] is False


def test_load_by_alias_and_version(tmp_path) -> None:
    reg = LocalModelRegistry(tmp_path)
    res = reg.maybe_promote(_detector(0.9))
    by_alias = reg.load("production")
    by_version = reg.load(res["version"])
    assert by_alias.model_version == by_version.model_version == res["version"]
