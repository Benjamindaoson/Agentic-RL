import json

import pytest

from scripts.compare_experiments import build_comparison, paired_stats, validate_paired
from scripts.verify_evidence import verify


def example(ids, values):
    rows = {
        key: {"success": value, "task": {
            "question": key, "db_id": "x", "evidence": "",
            "max_turns": 1, "context_limit": 4096,
        }}
        for key, value in zip(ids, values)
    }
    protocol = {
        "dataset_sha256": "abc", "task_ids_sha256": "def",
        "seed": 42, "temperature": 0, "tokenizer": "qwen",
        "budget": {"max_turns": 1},
    }
    metrics = {
        "model": "sql-policy", "policy_checkpoint": "x",
        "samples": len(rows), "task_accuracy": sum(values) / len(values),
        "first_turn_accuracy": 0.5, "mean_reward": 0.5,
        "mean_turns": 1.0, "invalid_sql_rate": 0.0,
        "latency_p95_ms": 100, "runner_error_count": 0,
        "trajectories_sha256": "hash",
    }
    return metrics, protocol, rows


def test_paired_compare_validates_budget_and_stats():
    a = example(["1", "2", "3", "4"], [False, False, True, True])
    b = example(["1", "2", "3", "4"], [True, True, True, True])
    validate_paired(a, b)
    diff = paired_stats(a[2], b[2], seed=42, n_boot=100)
    assert diff["gain_pp"] == 50.0
    assert diff["improved_tasks"] == 2
    assert diff["regressed_tasks"] == 0
    report = build_comparison({"base": a, "grpo": b}, n_boot=100)
    assert report["paired_vs_base"]["grpo"]["gain_pp"] == 50.0
    b[1]["budget"]["max_turns"] = 3
    with pytest.raises(ValueError, match="budget"):
        validate_paired(a, b)


def test_evidence_gate_fails_empty_run(tmp_path):
    result = verify(tmp_path / "train", tmp_path / "compare")
    assert result["passed"] is False
    assert result["problems"]
