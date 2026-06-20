from __future__ import annotations

from typing import Any

# Label Studio webhook actions that carry a new/updated human annotation.
ANNOTATION_ACTIONS = {"ANNOTATION_CREATED", "ANNOTATIONS_CREATED", "ANNOTATION_UPDATED"}


def parse_annotation_event(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Extract ``{event_id, review_fp_type, action}`` from a Label Studio webhook.

    Returns ``None`` for non-annotation actions or payloads missing an event id,
    so callers can ignore irrelevant webhook traffic.
    """
    action = payload.get("action")
    if action not in ANNOTATION_ACTIONS:
        return None

    annotation = payload.get("annotation") or {}
    task = payload.get("task") or annotation.get("task") or {}
    data = task.get("data", {}) if isinstance(task, dict) else {}
    event_id = data.get("event_id")
    if not event_id:
        return None

    review_fp_type = None
    for result in annotation.get("result", []):
        if result.get("from_name") == "fp_type":
            review_fp_type = (result.get("value", {}).get("choices") or [None])[0]
            break

    return {"event_id": event_id, "review_fp_type": review_fp_type, "action": action}


class ReviewBatcher:
    """Accumulate reviewed events with their labels and signal when a batch is ready.

    Debounces retraining so it fires once per ``threshold`` new annotations rather
    than on every single review. De-duplicates by event id, keeping the latest
    label so a re-reviewed event overwrites its earlier decision.
    """

    def __init__(self, threshold: int = 10) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self.threshold = threshold
        self._pending: dict[str, str | None] = {}

    def add(self, event_id: str, label: str | None = None) -> None:
        self._pending[event_id] = label

    @property
    def pending(self) -> int:
        return len(self._pending)

    def ready(self) -> bool:
        return self.pending >= self.threshold

    def drain(self) -> dict[str, str | None]:
        items = dict(self._pending)
        self._pending.clear()
        return items
