import json
import sqlite3

from scripts.filter_gold_sql import filter_manifest
from scripts.prepare_spider import main as prepare_spider_main
from scripts.validate_datasets import validate_spider


def test_gold_filter_preserves_all_ablations_and_exposes_exclusions(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("a", "b", "c"):
        folder = raw / "database" / name
        folder.mkdir(parents=True)
        with sqlite3.connect(folder / f"{name}.sqlite") as db:
            db.executescript("CREATE TABLE counts(n INTEGER); INSERT INTO counts VALUES(1),(2);")
    train = []
    for name in ("a", "b"):
        train.extend([
            {"db_id": name, "question": f"Count in {name}", "query": "SELECT COUNT(*) FROM counts"},
            {"db_id": name, "question": f"Invalid in {name}", "query": "SELECT x FROM no_such_table"},
        ])
    test = [{"db_id": "c", "question": "Sum?", "query": "SELECT SUM(n) FROM counts"}]
    (raw / "train_spider.json").write_text(json.dumps(train))
    (raw / "dev.json").write_text(json.dumps(test))
    prepared = tmp_path / "prepared"
    monkeypatch.setattr("sys.argv", [
        "prepare_spider.py", "--spider-root", str(raw), "--output-dir", str(prepared),
        "--internal-val-fraction", "0.5",
    ])
    prepare_spider_main()
    eligible = tmp_path / "eligible"
    result = filter_manifest(prepared / "manifest.json", eligible, max_rows=100000)
    assert result["unique_gold_tasks_checked"] == 5
    assert result["excluded_by_split"] == {"train": 1, "val": 1, "test": 0}
    assert result["full_official_dev_coverage"]
    assert result["coverage_ratio"]["train"] == 0.5
    assert len(result["exclusions"]) == 2
    verified = validate_spider(eligible / "manifest.json", verify_gold=True, max_rows=100000)
    assert verified["complete"]
    assert len(verified["variants"]) == 6
