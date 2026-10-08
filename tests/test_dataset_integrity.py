import copy
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.prepare_spider import main as prepare_spider_main
from scripts.validate_datasets import validate_bird, validate_spider


def _spider_fixture(tmp_path):
    root = tmp_path / "spider"
    root.mkdir()
    def make_db(name):
        folder = root / "database" / name
        folder.mkdir(parents=True)
        with sqlite3.connect(folder / f"{name}.sqlite") as conn:
            conn.executescript("CREATE TABLE items(id INTEGER PRIMARY KEY); INSERT INTO items VALUES (1),(2);")
    for name in ("a", "b", "c"):
        make_db(name)
    train = [{"db_id": "a", "question": "How many items?", "query": "SELECT COUNT(*) FROM items"},
             {"db_id": "b", "question": "Largest id?", "query": "SELECT MAX(id) FROM items"}]
    test = [{"db_id": "c", "question": "Smallest id?", "query": "SELECT MIN(id) FROM items"}]
    (root / "train_spider.json").write_text(json.dumps(train), encoding="utf-8")
    (root / "dev.json").write_text(json.dumps(test), encoding="utf-8")
    return root


def test_real_spider_three_way_preparation_and_full_gold_execution(tmp_path, monkeypatch):
    root = _spider_fixture(tmp_path)
    output = tmp_path / "prepared"
    monkeypatch.setattr("sys.argv", [
        "prepare_spider.py", "--spider-root", str(root),
        "--output-dir", str(output), "--internal-val-fraction", "0.5",
    ])
    prepare_spider_main()
    report = validate_spider(output / "manifest.json", verify_gold=True)
    assert report["complete"] and report["verified_gold_queries"]
    assert len(report["variants"]) == 6  # both context budgets include an optional self-check variant
    assert report["unique_gold_queries_executed"] == 3
    for variant in report["variants"]:
        assert all(part["gold_verified"] for part in variant["splits"].values())
        assert all(part["database_count"] == 1 for part in variant["splits"].values())


def test_spider_detects_manifest_or_data_tampering(tmp_path, monkeypatch):
    root = _spider_fixture(tmp_path)
    output = tmp_path / "prepared"
    monkeypatch.setattr("sys.argv", [
        "prepare_spider.py", "--spider-root", str(root),
        "--output-dir", str(output), "--internal-val-fraction", "0.5",
    ])
    prepare_spider_main()
    manifest_file = output / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(manifest)
    tampered["variants"][0]["samples"]["test"] += 1
    manifest_file.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="samples do not match"):
        validate_spider(manifest_file, verify_gold=False)
