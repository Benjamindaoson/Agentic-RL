#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-gpus", type=int, default=1)
    args = ap.parse_args()
    packages = ["torch", "agentlightning", "datasets", "sqlglot", "pandas", "pyarrow", "openai", "httpx"]
    report = {"python": sys.version.split()[0], "platform": platform.platform(), "packages": {}, "errors": []}
    loaded = {}
    for name in packages:
        try:
            mod = importlib.import_module(name)
            loaded[name] = mod
            report["packages"][name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            report["packages"][name] = None
            report["errors"].append(f"{name}: {exc}")
    for executable in ["agl-server", "agl-controller", "ray", "nvidia-smi"]:
        path = shutil.which(executable)
        report[executable] = path
        if not path:
            report["errors"].append(f"missing executable: {executable}")
    torch = loaded.get("torch")
    if torch is not None:
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        report["gpu_count"] = count
        report["gpus"] = [torch.cuda.get_device_name(i) for i in range(count)]
        if count < args.require_gpus:
            report["errors"].append(f"need {args.require_gpus} GPUs, found {count}")
    report["ok"] = not report["errors"]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
