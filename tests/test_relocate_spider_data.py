import hashlib
import json
import shutil
import sqlite3

import pytest

from agentic_rl_sql.dataset import read_training_parquet, unpack_task, write_parquet
from agentic_rl_sql.types import SqlTask
from scripts.relocate_spider_data import relocate
from scripts.validate_datasets import validate_spider


def setup_spider(tmp_path):
    db_root = tmp_path / "local_databases"
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    files, hashes = {}, {}
    for split, name in (("train", "a"), ("val", "b"), ("test", "c")):
        folder = db_root / name
        folder.mkdir(parents=True)
        db = folder / f"{name}.sqlite"
        with sqlite3.connect(db) as conn:
            conn.executescript("CREATE TABLE items(id INTEGER); INSERT INTO items VALUES(1);")
        task = SqlTask(
            f"spider-{split}-00000", "spider-1.0", split, name,
            str(db), "How many items?", "SELECT COUNT(*) FROM items",
            max_turns=1, context_limit=2048,
        )
        file = prepared / f"{split}_ctx2048_turn1.parquet"
        write_parquet([task], file)
        files[split] = str(file.resolve())
        hashes[split] = hashlib.sha256(json.dumps([name]).encode()).hexdigest()
    manifest = prepared / "manifest.json"
    manifest.write_text(json.dumps({
        "test_source": "official Spider 1.0 dev.json (Gold-eligible subset)",
        "variants": [{
            "name": "ctx2048_turn1", "context": 2048, "turns": 1,
            "explicit_check": False, "files": files,
            "samples": {"train":1, "val":1, "test":1},
            "database_counts": {"train":1, "val":1, "test":1},
            "database_ids_sha256": hashes,
        }],
    }), encoding="utf-8")
    (prepared / "gold_eligibility_audit.json").write_text(json.dumps({
        "verified_gold_eligible": True,
        "source_samples": {"train":1, "val":1, "test":1},
        "eligible_samples": {"train":1, "val":1, "test":1},
        "excluded_by_split": {"train":0, "val":0, "test":0},
        "coverage_ratio": {"train":1.0, "val":1.0, "test":1.0},
        "exclusions": [], "max_result_rows": 5000, "timeout_seconds": 8.0,
    }), encoding="utf-8")
    return manifest, db_root


def test_relocate_prepared_spider_to_new_host_and_recheck_gold(tmp_path):
    manifest, old_db_root = setup_spider(tmp_path)
    new_db_root = tmp_path / "gpu_mount"
    shutil.copytree(old_db_root, new_db_root)
    target = tmp_path / "relocated"
    result = relocate(manifest, new_db_root, target, verify_gold=True)
    assert result["status"] == "verified_relocation"
    assert result["gold_reexecuted"] is True
    assert len(result["relocated_files"]) == 3
    new_manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    new_train = new_manifest["variants"][0]["files"]["train"]
    task = unpack_task(read_training_parquet(new_train)[0])
    assert task.db_path.startswith(str(new_db_root.resolve()))
    assert validate_spider(target / "manifest.json", verify_gold=True)["complete"]
    original_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    old = unpack_task(read_training_parquet(original_manifest["variants"][0]["files"]["train"])[0])
    assert old.db_path.startswith(str(old_db_root.resolve()))
    with pytest.raises(FileExistsError):
        relocate(manifest, new_db_root, target)


def test_relocation_fails_if_any_database_is_missing(tmp_path):
    manifest, db_root = setup_spider(tmp_path)
    incomplete = tmp_path / "incomplete_databases"
    incomplete.mkdir()
    shutil.copytree(db_root / "a", incomplete / "a")
    with pytest.raises(FileNotFoundError, match="missing database"):
        relocate(manifest, incomplete, tmp_path / "output")
