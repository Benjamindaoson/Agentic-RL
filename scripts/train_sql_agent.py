#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any, cast

from datasets import Dataset as HuggingFaceDataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def default_overrides(args) -> dict[str, Any]:
    return {
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
        "data": {"train_batch_size": args.train_batch_size, "max_prompt_length": args.context_length, "max_response_length": args.max_response_length, "truncation": "error"},
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
                "fsdp_config": {"param_offload": args.param_offload, "optimizer_offload": args.optimizer_offload},
            },
            "ref": {"log_prob_micro_batch_size_per_gpu": args.reference_micro_batch_size, "fsdp_config": {"param_offload": True}},
            "model": {"path": args.model, "use_remove_padding": True, "enable_gradient_checkpointing": True},
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
        },
        "agentlightning": {
            "agl_base_url": args.agl_base_url,
            "agl_key": args.agl_key,
            "rollout_timeout_seconds": args.rollout_timeout,
            "async_rollout": {"enabled": args.async_mode, "async_train_batch_size": args.async_train_batch_size},
            "local": {
                "agent_class": "agent.sql_agent_entrypoint.Agent",
                "env_map": {"TASK_JSON": "input.task_json", "DB_PATH": "input.db_path", "MAX_TURNS": "input.max_turns", "DATASET_NAME": "input.dataset"},
            },
        },
    }


def build_config(args, dotlist: Sequence[str]) -> Any:
    import importlib.resources
    config_dir = str(importlib.resources.files("agentlightning.verl"))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        base = compose(config_name="config")
    OmegaConf.set_struct(base, False)
    return OmegaConf.merge(base, OmegaConf.create(default_overrides(args)), OmegaConf.from_dotlist(list(dotlist)))


def parse_args():
    ap = argparse.ArgumentParser(description="Train a self-correcting SQL policy with Agent Lightning + veRL GRPO.")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--run-name", default="qwen25_coder_3b_ctx4096_turn3")
    ap.add_argument("--agl-base-url", default="http://127.0.0.1:8181")
    ap.add_argument("--agl-key", default="agentic-rl-dev-key")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--context-length", type=int, default=4096)
    ap.add_argument("--max-response-length", type=int, default=2048)
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
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--save-freq", type=int, default=64)
    ap.add_argument("--test-freq", type=int, default=10)
    ap.add_argument("--rollout-timeout", type=int, default=300)
    ap.add_argument("--async", dest="async_mode", action="store_true")
    ap.add_argument("--async-train-batch-size", type=int, default=64)
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_known_args()


def main():
    args, dotlist = parse_args()
    from agentlightning.verl.entrypoint import run_ppo
    train_dataset: Sequence[Any] = cast(Sequence[Any], HuggingFaceDataset.from_parquet(args.train_file).to_list())
    val_dataset: Sequence[Any] = cast(Sequence[Any], HuggingFaceDataset.from_parquet(args.val_file).to_list())
    if not train_dataset or not val_dataset:
        raise SystemExit("train/validation data must be non-empty")
    config = build_config(args, dotlist)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(OmegaConf.to_yaml(config, resolve=True))
    run_ppo(config, train_dataset=train_dataset, val_dataset=val_dataset)


if __name__ == "__main__":
    main()
