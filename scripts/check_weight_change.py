#!/usr/bin/env python3
"""Compare actual exported HF tensors (not file hashes) across GRPO/no-update."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open


def tensor_index(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    result = {}
    for path in sorted(root.rglob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in result:
                    raise ValueError(f"duplicate tensor name {name} in {root}")
                result[name] = path
    if not result:
        raise ValueError(f"no safetensors in HF model directory: {root}")
    return result


def compare_weights(base_dir: Path, candidate_dir: Path, *, atol=1e-8, rtol=1e-6) -> dict:
    left, right = tensor_index(base_dir), tensor_index(candidate_dir)
    if set(left) != set(right):
        raise ValueError("checkpoint tensor names differ")
    total = changed = 0
    total_abs_delta = 0.0
    total_squared_delta = 0.0
    max_abs_delta = 0.0
    for name in sorted(left):
        with safe_open(left[name], framework="pt", device="cpu") as base_file:
            a = base_file.get_tensor(name)
        with safe_open(right[name], framework="pt", device="cpu") as candidate_file:
            b = candidate_file.get_tensor(name)
        if a.shape != b.shape:
            raise ValueError(f"tensor shape changed for {name}")
        if not a.is_floating_point():
            changed_here = torch.count_nonzero(a != b).item()
            error = (a.to(torch.float64) - b.to(torch.float64)).abs()
        else:
            error = (a.to(torch.float64) - b.to(torch.float64)).abs()
            threshold = atol + rtol * a.to(torch.float64).abs()
            changed_here = torch.count_nonzero(error > threshold).item()
        total += a.numel()
        changed += changed_here
        total_abs_delta += error.sum().item()
        total_squared_delta += torch.square(error).sum().item()
        max_abs_delta = max(max_abs_delta, error.max().item() if error.numel() else 0.0)
    return {
        "tensor_count": len(left), "element_count": total,
        "changed_elements": changed,
        "changed_fraction": changed / total if total else 0,
        "mean_absolute_delta": total_abs_delta / total if total else 0,
        "l2_delta": math.sqrt(total_squared_delta),
        "max_absolute_delta": max_abs_delta,
        "atol": atol, "rtol": rtol,
    }


def audit(base_dir: Path, grpo_dir: Path, no_update_dir: Path, **kw) -> dict:
    grpo = compare_weights(base_dir, grpo_dir, **kw)
    no_update = compare_weights(base_dir, no_update_dir, **kw)
    checks = {
        "grpo_policy_parameters_changed": grpo["changed_elements"] > 0,
        "no_update_policy_parameters_unchanged": no_update["changed_elements"] == 0,
    }
    return {"passed": all(checks.values()), "checks": checks, "grpo": grpo, "no_update": no_update}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-hf-dir", required=True)
    ap.add_argument("--grpo-hf-dir", required=True)
    ap.add_argument("--no-update-hf-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--rtol", type=float, default=1e-6)
    args = ap.parse_args()
    result = audit(
        Path(args.base_hf_dir), Path(args.grpo_hf_dir), Path(args.no_update_hf_dir),
        atol=args.atol, rtol=args.rtol,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
