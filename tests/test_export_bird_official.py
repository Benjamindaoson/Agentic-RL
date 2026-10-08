import hashlib
import json

import pytest

from scripts.export_bird_official import DELIMITER, export_predictions


def make_source(tmp_path, *, alter_question=False):
    data = tmp_path / "mini_dev_sqlite.json"
    records = [
        {"db_id": "music", "question": "How many songs?", "SQL": "SELECT count(*) FROM songs"},
        {"db_id": "books", "question": "Which book?", "SQL": "SELECT title FROM books"},
    ]
    data.write_text(json.dumps(records), encoding="utf-8")
    manifest = tmp_path / "bird_mini_dev_manifest.json"
    manifest.write_text(json.dumps({
        "source": "BIRD-SQL Mini-Dev official SQLite subset",
        "records_path": str(data),
        "records_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "total_supported_sql": 2, "skipped_non_select": 0,
        "skipped_missing_database": 0, "validated_gold_sql": True,
    }), encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    with predictions.open("w", encoding="utf-8") as sink:
        for idx, row in enumerate(records):
            sink.write(json.dumps({
                "task": {"task_id": f"bird-eval-{idx:05d}", "db_id": row["db_id"],
                         "question": ("wrong" if idx == 1 and alter_question else row["question"])},
                "final_sql": row["SQL"], "evaluation": {"protocol": "blind-final-v1"},
            }) + "\n")
    return manifest, predictions


def test_official_bird_export_preserves_canonical_order_and_db_id(tmp_path):
    manifest, predictions = make_source(tmp_path)
    out = tmp_path / "out"
    report = export_predictions(predictions, manifest, out, require_full_500=False)
    written = json.loads((out / "predicted_bird_minidev_sqlite.json").read_text(encoding="utf-8"))
    assert list(written) == ["0", "1"]
    assert written["0"] == "SELECT count(*) FROM songs" + DELIMITER + "music"
    assert written["1"] == "SELECT title FROM books" + DELIMITER + "books"
    assert report["official_ex_scored"] is False


def test_official_bird_export_rejects_misaligned_predictions(tmp_path):
    manifest, predictions = make_source(tmp_path, alter_question=True)
    with pytest.raises(ValueError, match="question mismatch"):
        export_predictions(predictions, manifest, tmp_path / "out", require_full_500=False)


def test_official_bird_export_rejects_corrupt_source(tmp_path):
    manifest, predictions = make_source(tmp_path)
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    original = metadata["records_path"]
    with open(original, "a", encoding="utf-8") as sink:
        sink.write("corruption")
    with pytest.raises(ValueError, match="records missing or have changed"):
        export_predictions(predictions, manifest, tmp_path / "out", require_full_500=False)
