import json
from pathlib import Path

import pytest

from scripts.export_bird_official import DELIMITER, run_official_ex


def test_upstream_official_bird_adapter_rejects_length_mismatch(tmp_path):
    repo = tmp_path / "bird-code"
    (repo / "evaluation").mkdir(parents=True)
    (repo / "evaluation" / "evaluation_ex.py").write_text("# placeholder")
    db_root = tmp_path / "db"
    db_root.mkdir()
    gold = tmp_path / "gold.sql"
    gold.write_text("SELECT 1\tmusic\n", encoding="utf-8")
    difficulty = tmp_path / "difficulty.jsonl"
    difficulty.write_text(json.dumps({"difficulty": "simple"}) + "\n", encoding="utf-8")
    predicted = tmp_path / "predictions.json"
    predicted.write_text(json.dumps({
        "0": "SELECT 1" + DELIMITER + "music",
        "1": "SELECT 2" + DELIMITER + "books",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="task lengths differ"):
        run_official_ex(
            {"prediction_path": str(predicted), "samples": 500,
             "official_full_500_rows": True,
             "full_official_predictions_are_ungraded": True},
            official_code=repo, db_root=db_root, gold_file=gold,
            difficulty_file=difficulty, timeout=1.0, cpus=1,
            output_dir=tmp_path,
        )
