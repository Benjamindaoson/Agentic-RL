import json
import sqlite3

from agentic_rl_sql.dataset import write_parquet
from agentic_rl_sql.types import SqlTask
from scripts.filter_bird_gold import filter_gold
from scripts.validate_datasets import validate_bird


def test_bird_gold_eligibility_discloses_invalid_sql(tmp_path):
    db = tmp_path / "mini.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript("CREATE TABLE x(n INTEGER); INSERT INTO x VALUES (1);")
    tasks = [
        SqlTask("b1", "bird", "eval", "mini", str(db), "Count?", "SELECT COUNT(*) FROM x"),
        SqlTask("b2", "bird", "eval", "mini", str(db), "Unscorable?", "SELECT * FROM no_table"),
    ]
    original = tmp_path / "original.parquet"
    write_parquet(tasks, original)
    original.with_suffix(".manifest.json").write_text(
        json.dumps({"samples": 2, "skipped_non_select": 0, "skipped_missing_database": 0})
    )
    target = tmp_path / "eligible.parquet"
    audit = filter_gold(original, target, timeout=3)
    assert audit["coverage_ratio"] == 0.5
    assert len(audit["excluded_gold"]) == 1
    assert not audit["comparable_to_full_official_mini_dev"]
    assert validate_bird(target, require_all_records=True)["complete"]
