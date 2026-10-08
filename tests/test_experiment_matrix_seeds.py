from pathlib import Path

import pytest
import yaml

from scripts.run_experiment_matrix import plan


def test_multi_seed_matrix_keeps_distinct_identity_and_cost_guard(tmp_path):
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/experiment_matrix.yaml").read_text())
    rows = plan(config, tmp_path / "prepared", tmp_path / "runs", "controls", seeds=[42, 123, 2026])
    assert len(rows) == 6
    assert len({r["name"] for r in rows}) == 6
    assert len({r["env"]["RUN_DIR"] for r in rows}) == 6
    assert sorted({r["seed"] for r in rows}) == [42, 123, 2026]
    for row in rows:
        assert "--seed" in row["command"]
        assert str(row["seed"]) in row["command"]
    with pytest.raises(ValueError, match="unique"):
        plan(config, tmp_path, tmp_path, "main", seeds=[42, 42])


def test_default_matrix_is_one_seed_and_does_not_launch_gpu(tmp_path):
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/experiment_matrix.yaml").read_text())
    rows = plan(config, tmp_path, tmp_path, "main")
    assert len(rows) == 1
    assert rows[0]["seed"] == 42
    assert rows[0]["name"] == "ctx4096_turn1"
