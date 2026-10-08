from agentic_rl_sql.dataset import read_training_parquet, unpack_task, write_parquet
from agentic_rl_sql.types import SqlTask


def test_parquet_roundtrip(tmp_path):
    task = SqlTask(
        task_id="x",
        dataset="toy",
        split="train",
        db_id="db",
        db_path="/tmp/db.sqlite",
        question="q",
        gold_sql="SELECT 1",
        max_turns=3,
    )
    path = tmp_path / "tasks.parquet"
    assert write_parquet([task], path) == 1
    records = read_training_parquet(path)
    restored = unpack_task(records[0])
    assert restored.task_id == task.task_id
    assert restored.gold_sql == "SELECT 1"
