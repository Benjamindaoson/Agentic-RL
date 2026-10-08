"""End-to-end CPU blind BIRD predictions -> official JSON adapter.

No GPU, no model benchmark: deterministic stub acts as a policy. Official
leaderboard scoring requires real model-generated predictions and official EX.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from agentic_rl_sql.dataset import write_parquet
from agentic_rl_sql.types import SqlTask
from scripts import run_blind_predictions
from scripts.export_bird_official import export_predictions


@pytest.mark.asyncio
async def test_blind_full_coverage_prediction_chain_without_gold(tmp_path, monkeypatch):
    db = tmp_path / "music.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "CREATE TABLE songs(id INTEGER); INSERT INTO songs VALUES(1),(2);"
        )
    raw_records = [
        {"db_id": "music", "question": "How many songs?",
         "SQL": "SELECT COUNT(*) FROM songs"},
        {"db_id": "music", "question": "Largest id?",
         "SQL": "SELECT MAX(id) FROM songs"},
    ]
    source = tmp_path / "mini_dev_sqlite.json"
    source.write_text(json.dumps(raw_records), encoding="utf-8")
    source_meta = tmp_path / "bird_mini_dev_manifest.json"
    source_meta.write_text(json.dumps({
        "source": "BIRD-SQL Mini-Dev official SQLite subset",
        "records_path": str(source),
        "records_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "total_supported_sql": 2, "skipped_non_select": 0,
        "skipped_missing_database": 0, "validated_gold_sql": True,
    }), encoding="utf-8")
    tasks = [
        SqlTask(f"bird-eval-{i:05d}", "bird-sql", "eval", "music", str(db),
                row["question"], row["SQL"], max_turns=1, context_limit=512)
        for i, row in enumerate(raw_records)
    ]
    data = tmp_path / "minidev.parquet"
    write_parquet(tasks, data)
    checkpoint = tmp_path / "policy_identity.json"
    checkpoint.write_text(json.dumps({"weights_fingerprint_sha256": "synthetic-test-only"}))
    class StubPolicy:
        def __init__(self, **kwargs):
            self.prompts = []
        async def complete(self, messages):
            self.prompts.append(messages)
            assert "gold_sql" not in json.dumps(messages)
            return '{"sql":"SELECT COUNT(*) FROM songs","decision":"final"}'
    monkeypatch.setattr(run_blind_predictions, "OpenAICompatibleClient", StubPolicy)
    monkeypatch.setattr(run_blind_predictions, "hf_message_counter", lambda model: lambda _: 60)
    args = SimpleNamespace(
        dataset=str(data), output=str(tmp_path / "predictions.jsonl"),
        model="mock-model", base_url="http://invalid", api_key="dummy",
        tokenizer="stub", policy_checkpoint="test", policy_manifest=str(checkpoint),
        seed=42, temperature=0.0, context_limit=512, max_turns=1,
        max_tokens=128, max_rows=100, sql_timeout=3.0, concurrency=2,
        require_official_mini_dev=False, resume=False,
    )
    result = await run_blind_predictions.run_blind(args)
    assert result["completed"] == 2
    assert result["official_ex_scored"] is False
    assert result["gold_access"] == "never"
    predicted = json.loads((tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert predicted["evaluation"]["protocol"] == "blind-ungraded-v1"
    assert "success" not in predicted and "reward" not in predicted
    final = export_predictions(tmp_path / "predictions.jsonl", source_meta,
                               tmp_path / "official", require_full_500=False)
    assert final["samples"] == 2
    assert final["official_ex_scored"] is False
    with pytest.raises(FileExistsError):
        await run_blind_predictions.run_blind(args)
    args.resume = True
    resumed = await run_blind_predictions.run_blind(args)
    assert resumed["completed"] == 2
