import json
import sqlite3

from scripts.prepare_bird_minidev import find_inputs, prepare


def test_official_style_minidev_structure_and_sql_execution(tmp_path):
    source = tmp_path / "canonical_archive"
    data = source / "mini_dev_data"
    folder = data / "dev_databases" / "mini"
    folder.mkdir(parents=True)
    db = folder / "mini.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE apples(id INTEGER)")
        conn.execute("INSERT INTO apples VALUES (1)")
    records = [
        {"db_id": "mini", "question": "How many apples?", "SQL": "SELECT COUNT(*) FROM apples", "evidence": ""},
        {"db_id": "mini", "question": "Which id?", "SQL": "SELECT id FROM apples", "evidence": ""},
    ]
    (data / "mini_dev_sqlite.json").write_text(json.dumps(records), encoding="utf-8")
    assert find_inputs(source)[0].is_file()
    output = tmp_path / "prepared"
    manifest = prepare(source, output, full_gold_audit=True)
    assert manifest["total_supported_sql"] == 2
    assert manifest["validated_gold_sql"]
    assert manifest["skipped_non_select"] == 0
