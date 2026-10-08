"""Independent BIRD model-eval preparation must preserve Gold subset coverage."""
from __future__ import annotations

import json
import sqlite3

import pytest

from agentic_rl_sql.dataset import read_training_parquet, write_parquet
from agentic_rl_sql.types import SqlTask
from scripts.evaluate_bird import verify_gold_eligible
from scripts.filter_bird_gold import filter_gold
from scripts.compare_experiments import validate_paired
from scripts.offline_readiness import audit


def _eligible(tmp_path, timeout=3.0, max_rows=123):
    db = tmp_path / "mini.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript("CREATE TABLE x(a INTEGER); INSERT INTO x VALUES(1),(2);")
    original = tmp_path / "bird.parquet"
    write_parquet([
        SqlTask("bird-1", "bird-sql", "eval", "mini", str(db),
                "count", "SELECT COUNT(*) FROM x"),
        SqlTask("bird-2", "bird-sql", "eval", "mini", str(db),
                "bad annotation", "SELECT z FROM missing"),
    ], original)
    original.with_suffix(".manifest.json").write_text(json.dumps({
        "samples": 2, "skipped_missing_database": 0, "skipped_non_select": 0,
    }))
    eligible = tmp_path / "bird_eligible.parquet"
    filter_gold(original, eligible, timeout=timeout, max_rows=max_rows)
    return eligible


def test_bird_model_evaluation_denominator_and_budgets(tmp_path):
    eligible = _eligible(tmp_path)
    score = verify_gold_eligible(eligible, timeout=3.0, max_rows=123)
    assert score["eligible_samples"] == 1
    assert score["excluded_count"] == 1
    assert len(read_training_parquet(eligible)) == 1
    with pytest.raises(ValueError, match="budgets|budget"):
        verify_gold_eligible(eligible, timeout=4, max_rows=123)
    with pytest.raises(ValueError, match="budgets|budget"):
        verify_gold_eligible(eligible, timeout=3.0, max_rows=5000)


def test_cpu_readiness_recognizes_real_prevalidated_bird_subset(tmp_path):
    eligible = _eligible(tmp_path)
    # Entire repository data gate requires the Spider manifest as well.
    result = audit(
        run_tests=False, require_datasets=True,
        eligible_manifest=tmp_path / "nonexistent_spider_manifest.json",
        bird_parquet=eligible,
    )
    assert result["checks"]["bird_minidev_gold_executable"] is True
    assert result["checks"]["spider_eligible_manifest_present"] is False
    assert result["passed"] is False


def test_paired_comparison_rejects_different_gold_subset_coverage():
    def group(coverage):
        rows = {"1": {"task": {"question": "q", "db_id": "x",
                               "evidence": "", "max_turns": 1,
                               "context_limit": 2048}, "success": True}}
        meta = {"runner_error_count": 0}
        proto = {
            "dataset_sha256": "same", "task_ids_sha256": "same",
            "seed": 42, "temperature": 0.0, "tokenizer": "same",
            "budget": {"max_turns": 1}, "gold_eligible_coverage": coverage,
        }
        return meta, proto, rows
    a = group({"eligible_test_fraction": 1.0})
    b = group({"eligible_test_fraction": 0.99})
    with pytest.raises(ValueError, match="gold_eligible_coverage"):
        validate_paired(a, b)
