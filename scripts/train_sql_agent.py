#!/usr/bin/env python3
"""Auditable Agent Lightning + veRL GRPO entrypoint; requires real GPU to train."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from datasets import Dataset as HuggingFaceDataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from agentic_rl_sql.dataset import unpack_task
from agentic_rl_sql.reward import load_reward_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_overrides(args) -> dict[str, Any]:
    return {
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
        "data": {
            "train_batch_size": args.train_batch_size,
            "seed": args.seed,
            "max_prompt_length": args.context_length,
            "max_response_length": args.max_response_length,
            "truncation": "error",
            "filter_overlong_prompts": False,
        },
        "actor_rollout_ref": {
            "rollout": {
                "tensor_model_parallel_size": args.tensor_parallel_size,
                "n": args.group_size,
                "log_prob_micro_batch_size_per_gpu": args.logprob_micro_batch_size,
                "multi_turn": {"format": "hermes"},
                "name": "vllm",
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "engine_kwargs": {"vllm": {"enable_auto_tool_choice": False}},
            },
            "actor": {
                "ppo_mini_batch_size": args.ppo_mini_batch_size,
                "ppo_micro_batch_size_per_gpu": args.ppo_micro_batch_size,
                "optim": {"lr": args.learning_rate},
                "use_kl_loss": args.use_kl_loss,
                "kl_loss_coef": args.kl_coefficient,
                "entropy_coeff": args.entropy_coefficient,
                "clip_ratio_low": args.clip_ratio_low,
                "clip_ratio_high": args.clip_ratio_high,
                "fsdp_config": {
                    "param_offload": args.param_offload,
                    "optimizer_offload": args.optimizer_offload,
                },
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": args.reference_micro_batch_size,
                "fsdp_config": {"param_offload": True},
            },
            "model": {
                "path": args.model,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": args.gpus,
            "val_before_train": True,
            "critic_warmup": 0,
            "logger": ["console", "wandb"] if args.wandb else ["console"],
            "project_name": "agentic-rl-sql",
            "experiment_name": args.run_name,
            "nnodes": 1,
            "save_freq": args.save_freq,
            "test_freq": args.test_freq,
            "total_epochs": args.epochs,
            "resume_mode": args.resume_mode,
            "default_local_dir": str(Path(args.run_dir).resolve() / "checkpoints"),
        },
        "agentlightning": {
            "agl_base_url": args.agl_base_url,
            "agl_key": args.agl_key,
            "rollout_timeout_seconds": args.rollout_timeout,
            "trace_aggregator": {
                "level": "trajectory",
                "trajectory_max_prompt_length": args.context_length,
                "trajectory_max_response_length": args.max_response_length,
            },
            "async_rollout": {
                "enabled": args.async_mode,
                "async_train_batch_size": args.async_train_batch_size,
            },
            "local": {
                "agent_class": "agent.sql_agent_entrypoint.Agent",
                "env_map": {
                    "TASK_JSON": "input.task_json",
                    "DB_PATH": "input.db_path",
                    "MAX_TURNS": "input.max_turns",
                },
            },
        },
    }


def build_config(args, dotlist: Sequence[str]) -> Any:
    """Compose the genuine installed veRL + Agent Lightning YAML hierarchy.

    Hydra's upstream pkg://verl.trainer.config searchpath imports verl.__init__,
    which eagerly imports CUDA-adjacent optional dependencies even when the
    operation is only CPU config validation. Loading the SAME packaged
    ppo_trainer.yaml group tree from its absolute path avoids that incidental
    import, and merging Agent Lightning's config.yaml with defaults removed
    reproduces upstream (ppo_trainer -> _self_) merge order.
    """
    import importlib.metadata
    import importlib.resources

    verl_root = Path(importlib.metadata.distribution("verl").locate_file("verl/trainer/config")).resolve()
    agl_root = Path(str(importlib.resources.files("agentlightning.verl"))).resolve()
    if not (verl_root / "ppo_trainer.yaml").is_file():
        raise FileNotFoundError(f"veRL distribution is missing ppo_trainer.yaml: {verl_root}")
    if not (agl_root / "config.yaml").is_file():
        raise FileNotFoundError(f"Agent Lightning distribution is missing config.yaml: {agl_root}")
    with initialize_config_dir(config_dir=str(verl_root), version_base=None):
        base = compose(config_name="ppo_trainer")
    overlay = OmegaConf.load(agl_root / "config.yaml")
    # The original AGL defaults are ["ppo_trainer", "_self_"]. Hydra and defaults
    # entries are parser directives, not runtime trainer configuration.
    if list(overlay.defaults) != ["ppo_trainer", "_self_"]:
        raise ValueError(f"Unexpected Agent Lightning config composition order: {overlay.defaults}")
    del overlay["defaults"]
    if "hydra" in overlay:
        del overlay["hydra"]
    OmegaConf.set_struct(base, False)
    config = OmegaConf.merge(
        base, overlay, OmegaConf.create(default_overrides(args)),
        OmegaConf.from_dotlist(list(dotlist)),
    )
    if int(config.data.max_prompt_length) != int(config.agentlightning.trace_aggregator.trajectory_max_prompt_length):
        raise ValueError("prompt length and trajectory aggregator prompt budget must match")
    if int(config.data.max_response_length) != int(config.agentlightning.trace_aggregator.trajectory_max_response_length):
        raise ValueError("response length and trajectory aggregator response budget must match")
    if str(config.algorithm.adv_estimator).lower() != "grpo":
        raise ValueError("this project requires GRPO; do not override the advantage estimator")
    return config

def validate_training_dataset(train_rows, val_rows, args) -> dict:
    if not train_rows or not val_rows:
        raise ValueError("training and validation datasets must be nonempty")
    train_tasks = [unpack_task(r) for r in train_rows]
    val_tasks = [unpack_task(r) for r in val_rows]
    for task in train_tasks + val_tasks:
        if task.context_limit != args.context_length:
            raise ValueError(f"context budget mismatch in task {task.task_id}")
        if task.max_turns != args.max_turns:
            raise ValueError(f"turn budget mismatch in task {task.task_id}")
        if bool(task.metadata.get("explicit_check")) != args.explicit_check:
            raise ValueError(f"explicit check mismatch in task {task.task_id}")
        if not Path(task.db_path).is_file():
            raise FileNotFoundError(task.db_path)
    train_ids = {t.task_id for t in train_tasks}
    val_ids = {t.task_id for t in val_tasks}
    if len(train_ids) != len(train_tasks) or len(val_ids) != len(val_tasks):
        raise ValueError("duplicate task ids")
    if train_ids & val_ids:
        raise ValueError("train and validation task IDs overlap")
    train_schemas = {t.db_id for t in train_tasks}
    val_schemas = {t.db_id for t in val_tasks}
    overlap = train_schemas & val_schemas
    if overlap and not args.allow_schema_overlap:
        raise ValueError(f"train/validation database overlap: {sorted(overlap)[:10]}")
    return {
        "train_samples": len(train_tasks), "val_samples": len(val_tasks),
        "train_db_count": len(train_schemas), "val_db_count": len(val_schemas),
        "schema_overlap_count": len(overlap),
        "allow_schema_overlap": args.allow_schema_overlap,
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Train a blind SQL policy with RLVR + GRPO.")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--base-model-revision", default="", help="Pinned immutable model revision for provenance")
    ap.add_argument("--run-name", default="qwen25_coder_3b_ctx4096_turn1")
    ap.add_argument("--run-dir", default="runs/grpo")
    ap.add_argument("--agl-base-url", default="http://127.0.0.1:8181")
    ap.add_argument("--agl-key", default="agentic-rl-dev-key")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--context-length", type=int, default=4096)
    ap.add_argument("--max-response-length", type=int, default=1024)
    ap.add_argument("--max-turns", type=int, default=1)
    ap.add_argument("--explicit-check", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--train-batch-size", type=int, default=32)
    ap.add_argument("--ppo-mini-batch-size", type=int, default=32)
    ap.add_argument("--ppo-micro-batch-size", type=int, default=4)
    ap.add_argument("--logprob-micro-batch-size", type=int, default=4)
    ap.add_argument("--reference-micro-batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--clip-ratio-low", type=float, default=0.2)
    ap.add_argument("--clip-ratio-high", type=float, default=0.3)
    ap.add_argument("--use-kl-loss", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--kl-coefficient", type=float, default=0.001)
    ap.add_argument("--entropy-coefficient", type=float, default=0.0)
    ap.add_argument("--param-offload", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--optimizer-offload", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--save-freq", type=int, default=1)
    ap.add_argument("--test-freq", type=int, default=1)
    ap.add_argument("--resume-mode", choices=["disable", "auto"], default="disable")
    ap.add_argument("--rollout-timeout", type=int, default=300)
    ap.add_argument("--async", dest="async_mode", action="store_true")
    ap.add_argument("--async-train-batch-size", type=int, default=64)
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--reward-config", default="configs/reward.yaml")
    ap.add_argument("--reward-mode", choices=["execution", "validity_only"], default="execution")
    ap.add_argument("--allow-schema-overlap", action="store_true", help="toy/testing ONLY; forbidden for Spider")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_known_args()


def main():
    args, dotlist = parse_args()
    if args.group_size < 2 or args.gpus < 1 or args.learning_rate < 0:
        raise ValueError("invalid GRPO group/gpu/learning-rate")
    if args.allow_schema_overlap and "spider" in args.train_file.lower():
        raise ValueError("Spider must remain cross-schema")
    train_path, val_path = Path(args.train_file).resolve(), Path(args.val_file).resolve()
    rows = {
        "train": cast(Sequence[Any], HuggingFaceDataset.from_parquet(str(train_path)).to_list()),
        "val": cast(Sequence[Any], HuggingFaceDataset.from_parquet(str(val_path)).to_list()),
    }
    dataset_summary = validate_training_dataset(rows["train"], rows["val"], args)
    reward_cfg = load_reward_config(args.reward_config, args.reward_mode)
    config = build_config(args, dotlist)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    # Set runtime contract on the controller/worker environment.
    os.environ["REWARD_CONFIG"] = str(Path(args.reward_config).resolve())
    os.environ["REWARD_MODE"] = args.reward_mode
    os.environ["POLICY_PROMPT_TOKEN_LIMIT"] = str(args.context_length)
    os.environ["POLICY_TOKENIZER_PATH"] = args.model
    os.environ["ROLLOUT_MAX_TOKENS"] = str(args.max_response_length)
    os.environ["ROLLOUT_TEMPERATURE"] = os.environ.get("ROLLOUT_TEMPERATURE", "0.7")
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    config_for_disk = OmegaConf.to_container(config, resolve=True)
    config_for_disk["agentlightning"]["agl_key"] = "[REDACTED]"
    (run_dir / "training_config.json").write_text(
        json.dumps(config_for_disk, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "status": "launched_not_verified",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit, "run_name": args.run_name, "algorithm": "GRPO",
        "base_model": args.model,
        "base_model_revision": args.base_model_revision or "LOCAL_WEIGHT_HASH" if (run_dir / "base_model_identity.json").exists() else "UNPINNED",
        "base_model_identity_file": "base_model_identity.json" if (run_dir / "base_model_identity.json").exists() else None,
        "seed": args.seed, "learning_rate": args.learning_rate,
        "reward_mode": reward_cfg.mode, "reward_config": reward_cfg.to_dict(),
        "reward_config_sha256": sha256_file(Path(args.reward_config)),
        "train_file": str(train_path), "train_sha256": sha256_file(train_path),
        "val_file": str(val_path), "val_sha256": sha256_file(val_path),
        "datasets": dataset_summary,
        "context_limit": args.context_length, "max_response_length": args.max_response_length,
        "max_turns": args.max_turns, "explicit_check": args.explicit_check,
        "group_size": args.group_size,
        "eval_protocol": "blind-final-v1",
        "checkpoint_root": str(run_dir / "checkpoints"),
        "training_config_file": "training_config.json",
    }
    manifest_path = run_dir / "grpo_run_manifest.json"
    if manifest_path.exists():
        if args.resume_mode != "auto":
            raise RuntimeError("run already exists; use a fresh name or --resume-mode auto")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        locked = (
            "base_model", "base_model_revision", "train_sha256", "val_sha256",
            "context_limit", "max_response_length", "max_turns", "explicit_check",
            "reward_config_sha256", "reward_mode", "group_size", "seed", "learning_rate",
        )
        changes = [key for key in locked if previous.get(key) != manifest.get(key)]
        if changes:
            raise RuntimeError(f"resume would mix incompatible experiment settings: {changes}")
        manifest["resume_from_status"] = previous.get("status")
    elif args.resume_mode == "auto":
        raise RuntimeError("resume requested but no previous run manifest exists")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(dataset_summary, indent=2))
    print("Training configuration written to", run_dir)
    from agentlightning.verl.entrypoint import run_ppo
    run_ppo(config, train_dataset=rows["train"], val_dataset=rows["val"])
    # Not a claim of success until checkpoint + logs are independently audited.
    manifest["status"] = "trainer_returned_verify_metrics_and_checkpoints"
    (run_dir / "grpo_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
