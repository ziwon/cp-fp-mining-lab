from cv_fp_lab.gating import evaluate_promotion


def _m(map50, recall, neg_fp_rate, rpc=None):
    d = {"map50": map50, "recall": recall, "neg_fp_rate": neg_fp_rate}
    if rpc is not None:
        d["recall_per_class"] = rpc
    return d


def test_first_model_promotes() -> None:
    res = evaluate_promotion(_m(0.5, 0.5, 0.2), None)
    assert res["promoted"] is True
    assert "first model" in res["reason"]


def test_better_candidate_promotes() -> None:
    prod = _m(0.70, 0.70, 0.30)
    cand = _m(0.77, 0.71, 0.20)  # higher mAP, higher recall, lower FP-rate
    res = evaluate_promotion(cand, prod)
    assert res["promoted"] is True
    assert all(c["passed"] for c in res["checks"].values())


def test_map_regression_blocks() -> None:
    prod = _m(0.77, 0.70, 0.20)
    cand = _m(0.70, 0.70, 0.20)  # mAP dropped well beyond tolerance
    res = evaluate_promotion(cand, prod)
    assert res["promoted"] is False
    assert res["checks"]["map50"]["passed"] is False


def test_fp_rate_increase_blocks() -> None:
    prod = _m(0.77, 0.70, 0.20)
    cand = _m(0.78, 0.71, 0.30)  # better mAP/recall but more false positives
    res = evaluate_promotion(cand, prod, fp_rate_max_delta=0.0)
    assert res["promoted"] is False
    assert res["checks"]["neg_fp_rate"]["passed"] is False


def test_per_class_recall_floor_blocks() -> None:
    prod = _m(0.77, 0.70, 0.20, rpc={"smoke": 0.70, "fire": 0.70})
    cand = _m(0.78, 0.70, 0.18, rpc={"smoke": 0.71, "fire": 0.55})  # fire recall collapses
    res = evaluate_promotion(cand, prod, recall_min_delta=-0.02)
    assert res["promoted"] is False
    assert res["checks"]["recall[fire]"]["passed"] is False
    assert res["checks"]["recall[smoke]"]["passed"] is True


def test_tolerance_allows_small_regression() -> None:
    prod = _m(0.770, 0.70, 0.20)
    cand = _m(0.765, 0.70, 0.20)  # -0.005, within map50_min_delta=-0.01
    res = evaluate_promotion(cand, prod, map50_min_delta=-0.01)
    assert res["promoted"] is True
