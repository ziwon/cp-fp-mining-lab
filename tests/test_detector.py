import numpy as np

from cv_fp_lab.detector import FpDetector


def _separable_data(n_per=20, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=[0, 0, 0], scale=0.1, size=(n_per, 3))
    b = rng.normal(loc=[5, 5, 5], scale=0.1, size=(n_per, 3))
    c = rng.normal(loc=[0, 5, 0], scale=0.1, size=(n_per, 3))
    x = np.vstack([a, b, c])
    y = np.array(["steam"] * n_per + ["fog"] * n_per + ["shadow"] * n_per)
    return x, y


def test_train_predicts_separable_classes() -> None:
    x, y = _separable_data()
    det = FpDetector.train(x, y)
    assert set(det.classes) == {"steam", "fog", "shadow"}
    assert det.metrics["macro_f1"] > 0.9
    assert (det.predict(x) == y).mean() > 0.9


def test_uncertainty_is_bounded_and_higher_for_ambiguous() -> None:
    x, y = _separable_data()
    det = FpDetector.train(x, y)
    confident = det.uncertainty(np.array([[0, 0, 0]]))
    ambiguous = det.uncertainty(np.array([[2.5, 2.5, 2.5]]))  # between clusters
    assert 0.0 <= confident[0] <= 1.0
    assert ambiguous[0] > confident[0]


def test_save_load_roundtrip(tmp_path) -> None:
    x, y = _separable_data()
    det = FpDetector.train(x, y)
    path = det.save(tmp_path / "m.joblib")
    loaded = FpDetector.load(path)
    assert loaded.model_version == det.model_version
    assert loaded.classes == det.classes
    np.testing.assert_array_equal(loaded.predict(x), det.predict(x))
