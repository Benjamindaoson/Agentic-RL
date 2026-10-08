import json

import pytest

from scripts.checkpoint_manifest import build_manifest
from scripts.extract_training_metrics import parse_metrics
from scripts.train_sql_agent import default_overrides


def test_training_context_aggregation_matches_budget():
    class Args:
        train_batch_size = 32
        context_length = 4096
        max_response_length = 1024
        tensor_parallel_size = 1
        group_size = 4
        logprob_micro_batch_size = 4
        gpu_memory_utilization = 0.65
        ppo_mini_batch_size = 32
        ppo_micro_batch_size = 4
        learning_rate = 1e-6
        use_kl_loss = True
        kl_coefficient = 0.001
        entropy_coefficient = 0
        clip_ratio_low = 0.2
        clip_ratio_high = 0.3
        param_offload = True
        optimizer_offload = True
        reference_micro_batch_size = 8
        model = "test"
        gpus = 1
        wandb = False
        run_name = "test"
        save_freq = 1
        test_freq = 1
        epochs = 1
        resume_mode = "disable"
        run_dir = "runs/test"
        agl_base_url = "http://127.0.0.1:8181"
        agl_key = "test"
        rollout_timeout = 300
        async_mode = False
        async_train_batch_size = 64
    cfg = default_overrides(Args())
    assert cfg["agentlightning"]["trace_aggregator"]["trajectory_max_prompt_length"] == 4096
    assert cfg["data"]["max_prompt_length"] == 4096
    assert cfg["actor_rollout_ref"]["rollout"]["n"] == 4


def test_extract_real_console_step_metrics():
    lines = [
        "garbage info",
        "step:1 - actor/pg_loss:0.112 - actor/kl_loss:0.001 - training/global_step:1",
        '{"step":2,"actor/pg_loss":0.2,"training/n_groups":4}',
    ]
    records = list(parse_metrics(lines))
    assert len(records) == 2
    assert records[0]["global_step"] == 1
    assert records[1]["actor/pg_loss"] == 0.2


def test_checkpoint_manifest_requires_weights(tmp_path):
    with pytest.raises(ValueError):
        build_manifest(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"faux checkpoint shard for hash test")
    manifest = build_manifest(tmp_path)
    assert manifest["weight_file_count"] == 1
    assert manifest["files"]["model.safetensors"]["sha256"]
