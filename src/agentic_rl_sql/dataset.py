from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .types import SqlTask


def task_to_training_record(task: SqlTask) -> dict[str, Any]:
    return {
        "input": {
            "task_json": json.dumps(task.to_dict(), ensure_ascii=False),
            "db_path": task.db_path,
            "max_turns": str(task.max_turns),
            "dataset": task.dataset,
        },
        "metadata": {
            "task_id": task.task_id,
            "db_id": task.db_id,
            "split": task.split,
            "context_limit": task.context_limit,
        },
    }


def write_parquet(tasks: Iterable[SqlTask], path: str | Path) -> int:
    rows = [task_to_training_record(task) for task in tasks]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    return len(rows)


def read_training_parquet(path: str | Path) -> list[dict[str, Any]]:
    return pd.read_parquet(path).to_dict(orient="records")


def unpack_task(record: dict[str, Any]) -> SqlTask:
    input_data = record.get("input", record)
    raw = input_data.get("task_json") if isinstance(input_data, dict) else None
    if not raw:
        raise ValueError("record has no input.task_json")
    return SqlTask.from_dict(json.loads(raw))
