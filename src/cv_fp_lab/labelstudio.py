from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


LABEL_CONFIG = """
<View>
  <Image name="image" value="$image"/>
  <Header value="Model prediction: $pred_class / confidence: $pred_confidence"/>
  <Choices name="is_event" toName="image" choice="single" required="true">
    <Choice value="real_event"/>
    <Choice value="false_positive"/>
    <Choice value="uncertain"/>
  </Choices>
  <Choices name="fp_type" toName="image" choice="single">
    <Choice value="steam"/>
    <Choice value="fog"/>
    <Choice value="dust"/>
    <Choice value="reflection"/>
    <Choice value="headlight"/>
    <Choice value="shadow"/>
    <Choice value="sitting"/>
    <Choice value="animal"/>
    <Choice value="authorized_worker"/>
    <Choice value="unknown"/>
  </Choices>
  <Choices name="bbox_valid" toName="image" choice="single">
    <Choice value="valid"/>
    <Choice value="wrong_class"/>
    <Choice value="wrong_location"/>
    <Choice value="unnecessary"/>
  </Choices>
  <TextArea name="comment" toName="image" placeholder="Root cause or reviewer comment"/>
</View>
""".strip()


def dataframe_to_labelstudio_tasks(df: pd.DataFrame, output_path: str | Path) -> list[dict[str, Any]]:
    tasks = []
    for row in df.to_dict(orient="records"):
        data = {
            "image": row["image_path"],
            "event_id": row["event_id"],
            "camera_id": row["camera_id"],
            "site_id": row["site_id"],
            "pred_class": row["pred_class"],
            "pred_confidence": row["pred_confidence"],
            "cluster_id": int(row.get("cluster_id", -1)),
        }
        # Active-learning fields are optional; include them so reviewers can sort
        # tasks by informativeness in Label Studio when ranking has been applied.
        if "uncertainty" in row and pd.notna(row["uncertainty"]):
            data["uncertainty"] = round(float(row["uncertainty"]), 4)
        if "acquisition_rank" in row and pd.notna(row["acquisition_rank"]):
            data["acquisition_rank"] = int(row["acquisition_rank"])
        tasks.append(
            {
                "data": data,
                "predictions": [
                    {
                        "model_version": row.get("model_version", "unknown"),
                        "score": row.get("pred_confidence", 0.0),
                        "result": [],
                    }
                ],
            }
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    return tasks


def parse_labelstudio_export(path: str | Path) -> pd.DataFrame:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for task in raw:
        data = task.get("data", {})
        row = {
            "event_id": data.get("event_id"),
            "image_path": data.get("image"),
            "camera_id": data.get("camera_id"),
            "site_id": data.get("site_id"),
            "pred_class": data.get("pred_class"),
            "pred_confidence": data.get("pred_confidence"),
            "cluster_id": data.get("cluster_id"),
            "review_is_event": None,
            "review_fp_type": None,
            "review_bbox_valid": None,
            "review_comment": None,
        }
        annotations = task.get("annotations", [])
        results = annotations[0].get("result", []) if annotations else []
        for result in results:
            name = result.get("from_name")
            value = result.get("value", {})
            if name == "is_event":
                row["review_is_event"] = (value.get("choices") or [None])[0]
            elif name == "fp_type":
                row["review_fp_type"] = (value.get("choices") or [None])[0]
            elif name == "bbox_valid":
                row["review_bbox_valid"] = (value.get("choices") or [None])[0]
            elif name == "comment":
                row["review_comment"] = (value.get("text") or [None])[0]
        rows.append(row)
    return pd.DataFrame(rows)
