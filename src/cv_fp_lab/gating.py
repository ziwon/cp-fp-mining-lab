from __future__ import annotations

from typing import Any


def evaluate_promotion(
    candidate: dict[str, Any],
    production: dict[str, Any] | None,
    map50_min_delta: float = -0.01,
    recall_min_delta: float = -0.02,
    fp_rate_max_delta: float = 0.0,
) -> dict[str, Any]:
    """Detection-aware, multi-criteria promotion gate.

    Promotes a candidate only if it clears *every* check against production:

    - **map50**: ``candidate >= production + map50_min_delta`` (no quality regression)
    - **recall** (per class when available, else overall):
      ``candidate >= production + recall_min_delta`` (keep detecting real events)
    - **fp_rate** (negative-image FP-rate): ``candidate <= production + fp_rate_max_delta``
      (do not increase false positives — the point of FP mining is to lower this)

    Deltas are tolerances: negative allows a small regression, ``0.0`` is strict.
    With no production model the candidate is promoted as the first model. Returns
    a structured decision with per-check pass/fail and human-readable reasons.
    """
    if production is None:
        return {
            "promoted": True,
            "reason": "first model (no production to compare)",
            "checks": {},
        }

    checks: dict[str, dict[str, Any]] = {}

    def _check(name: str, cand: float, prod: float, *, lower_is_better: bool, delta: float) -> bool:
        if lower_is_better:
            ok = cand <= prod + delta
            rel = "<=" if ok else ">"
            detail = f"{cand:.4f} {rel} {prod:.4f} + {delta}"
        else:
            ok = cand >= prod + delta
            rel = ">=" if ok else "<"
            detail = f"{cand:.4f} {rel} {prod:.4f} + {delta}"
        checks[name] = {"passed": ok, "candidate": cand, "production": prod, "detail": detail}
        return ok

    _check("map50", candidate["map50"], production["map50"], lower_is_better=False, delta=map50_min_delta)

    # Per-class recall when both sides expose it; otherwise overall recall.
    cand_rpc = candidate.get("recall_per_class") or {}
    prod_rpc = production.get("recall_per_class") or {}
    shared = set(cand_rpc) & set(prod_rpc)
    if shared:
        for cls in sorted(shared):
            _check(
                f"recall[{cls}]", cand_rpc[cls], prod_rpc[cls],
                lower_is_better=False, delta=recall_min_delta,
            )
    else:
        _check("recall", candidate["recall"], production["recall"], lower_is_better=False, delta=recall_min_delta)

    _check(
        "neg_fp_rate", candidate["neg_fp_rate"], production["neg_fp_rate"],
        lower_is_better=True, delta=fp_rate_max_delta,
    )

    failed = [name for name, c in checks.items() if not c["passed"]]
    promoted = not failed
    reason = "all checks passed" if promoted else "failed: " + ", ".join(failed)
    return {"promoted": promoted, "reason": reason, "checks": checks}
