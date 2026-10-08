#!/usr/bin/env python3
"""CPU-only contract with *installed* Agent Lightning 1.0.2 and veRL 0.8.0.

Hydra composes actual upstream configuration files; AGL's event schema checks
our payload. No optimizer, CUDA, vLLM or model forward pass is executed.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import importlib.resources
import json
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

PINNED = {"agentlightning": "1.0.2", "verl": "0.8.0"}


def _function_signature_from_source(package: str, filename: str, name: str) -> list[str]:
    source = (importlib.resources.files(package) / filename).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return [arg.arg for arg in node.args.args]
    raise ValueError(f"{package}/{filename} is missing expected function {name}")


def contract() -> dict:
    found = {key: importlib.metadata.version(key) for key in PINNED}
    if found != PINNED:
        raise RuntimeError(f"unsupported framework version combination: {found}, expected {PINNED}")
    # Avoid loading Ray/CUDA modules; inspect actual installed upstream source.
    run_ppo_args = _function_signature_from_source("agentlightning.verl", "entrypoint.py", "run_ppo")
    if run_ppo_args[:3] != ["config", "train_dataset", "val_dataset"]:
        raise RuntimeError(f"AGL run_ppo signature drift: {run_ppo_args}")
    from scripts.train_sql_agent import build_config, parse_args

    composed = {}
    for limit in (2048, 4096):
        with patch("sys.argv", ["train_sql_agent.py", "--train-file", "dummy",
                                "--val-file", "dummy", "--context-length", str(limit)]):
            args, remaining = parse_args()
        cfg = build_config(args, remaining)
        # The exact upstream Hydra hierarchy matters more than local dictionary tests.
        rendered = OmegaConf.to_container(cfg, resolve=True)
        assert rendered["algorithm"]["adv_estimator"] == "grpo"
        assert rendered["data"]["max_prompt_length"] == limit
        assert rendered["agentlightning"]["trace_aggregator"]["level"] == "trajectory"
        assert rendered["agentlightning"]["trace_aggregator"]["trajectory_max_prompt_length"] == limit
        assert rendered["actor_rollout_ref"]["rollout"]["n"] == 4
        assert rendered["actor_rollout_ref"]["actor"]["use_kl_loss"] is True
        assert rendered["agentlightning"]["local"]["agent_class"] == "agent.sql_agent_entrypoint.Agent"
        composed[str(limit)] = {
            "prompt_budget": rendered["data"]["max_prompt_length"],
            "trajectory_prompt_budget": rendered["agentlightning"]["trace_aggregator"]["trajectory_max_prompt_length"],
            "algorithm": rendered["algorithm"]["adv_estimator"],
            "group_size": rendered["actor_rollout_ref"]["rollout"]["n"],
        }
    module = importlib.import_module("agent.sql_agent_entrypoint")
    if not hasattr(module.Agent, "run"):
        raise RuntimeError("Agent Lightning local runner path is invalid")

    # Checks actual installed schema and catches invalid AGL event envelopes.
    from agentlightning.schemas import EventCreate
    event = EventCreate.model_validate({
        "event_type": "reward",
        "data": {"value": 0.9, "metadata": {"task_id": "cpu-contract"}},
    })
    if event.event_type != "reward" or event.data["value"] != 0.9:
        raise AssertionError("upstream Agent Lightning reward event contract mismatch")
    return {
        "passed": True, "scope": "CPU upstream import/Hydra/AGL reward-envelope contract",
        "framework_versions": found, "agl_run_ppo_args": run_ppo_args,
        "composed_experiments": composed,
        "gpu_training_performed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = contract()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
