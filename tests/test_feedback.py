from cv_fp_lab.feedback import ReviewBatcher, parse_annotation_event


def _event(action="ANNOTATION_CREATED", event_id="evt_1", fp_type="steam"):
    result = []
    if fp_type is not None:
        result = [{"from_name": "fp_type", "to_name": "image", "value": {"choices": [fp_type]}}]
    return {
        "action": action,
        "annotation": {"result": result},
        "task": {"data": {"event_id": event_id}},
    }


def test_parse_extracts_event_and_label() -> None:
    parsed = parse_annotation_event(_event())
    assert parsed == {
        "event_id": "evt_1",
        "review_fp_type": "steam",
        "action": "ANNOTATION_CREATED",
    }


def test_parse_ignores_non_annotation_actions() -> None:
    assert parse_annotation_event(_event(action="PROJECT_UPDATED")) is None


def test_parse_requires_event_id() -> None:
    payload = _event()
    payload["task"]["data"] = {}
    assert parse_annotation_event(payload) is None


def test_batcher_dedupes_and_signals_threshold() -> None:
    b = ReviewBatcher(threshold=3)
    b.add("a", "steam")
    b.add("a", "fog")  # re-review overwrites the label
    b.add("b", "shadow")
    assert b.ready() is False
    b.add("c", "animal")
    assert b.ready() is True
    drained = b.drain()
    assert drained == {"a": "fog", "b": "shadow", "c": "animal"}
    assert b.pending == 0
    assert b.ready() is False
