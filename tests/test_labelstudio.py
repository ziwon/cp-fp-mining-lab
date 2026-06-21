import json

import pandas as pd

from cv_fp_lab.labelstudio import dataframe_to_labelstudio_tasks, parse_labelstudio_export


def test_labelstudio_task_export_and_review_parse(tmp_path) -> None:
    tasks_path = tmp_path / "tasks.json"
    df = pd.DataFrame(
        [
            {
                "event_id": "evt_1",
                "image_path": "data/raw/evt_1.png",
                "camera_id": "cam-001",
                "site_id": "site-01",
                "pred_class": "smoke",
                "pred_confidence": 0.9,
                "cluster_id": 3,
                "synthetic_fp_type": "steam",
                "model_version": "detector-test",
            }
        ]
    )

    tasks = dataframe_to_labelstudio_tasks(df, tasks_path)

    assert tasks_path.exists()
    assert tasks[0]["data"]["cluster_id"] == 3
    assert tasks[0]["predictions"][0]["model_version"] == "detector-test"
    assert tasks[0]["predictions"][0]["result"][0]["value"]["choices"] == ["steam"]

    tasks[0]["annotations"] = [
        {
            "result": [
                {"from_name": "is_event", "value": {"choices": ["false_positive"]}},
                {"from_name": "fp_type", "value": {"choices": ["steam"]}},
                {"from_name": "bbox_valid", "value": {"choices": ["wrong_class"]}},
                {"from_name": "comment", "value": {"text": ["review note"]}},
            ]
        }
    ]
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(tasks), encoding="utf-8")

    reviewed = parse_labelstudio_export(export_path)

    assert reviewed.loc[0, "event_id"] == "evt_1"
    assert reviewed.loc[0, "review_is_event"] == "false_positive"
    assert reviewed.loc[0, "review_fp_type"] == "steam"
    assert reviewed.loc[0, "review_bbox_valid"] == "wrong_class"
    assert reviewed.loc[0, "review_comment"] == "review note"
