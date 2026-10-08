from agentic_rl_sql.types import SqlTask
from scripts.prepare_spider import grouped_split, validate_disjoint


def test_split_is_by_database_not_question():
    tasks = [
        SqlTask(f"spider-train-{i:05d}", "spider", "train", f"db{i // 3}",
                "/tmp/irrelevant.sqlite", "question", "SELECT 1")
        for i in range(30)
    ]
    train, val = grouped_split(tasks, seed=42, val_fraction=0.2)
    assert train and val
    assert {t.db_id for t in train}.isdisjoint({t.db_id for t in val})
    assert all(t.split == "val" and t.task_id.startswith("spider-val-") for t in val)
    test = [SqlTask("test", "spider", "test", "external",
                    "/tmp/irrelevant.sqlite", "question", "SELECT 1")]
    validate_disjoint(train, val, test)


def test_group_split_is_deterministic():
    def tasks():
        return [
            SqlTask(f"spider-train-{i}", "spider", "train", f"db{i}",
                    "/tmp/db", "question", "SELECT 1")
            for i in range(20)
        ]
    a, b = grouped_split(tasks(), seed=42), grouped_split(tasks(), seed=42)
    assert {x.db_id for x in a[1]} == {x.db_id for x in b[1]}
