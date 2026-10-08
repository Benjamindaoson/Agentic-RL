from __future__ import annotations

import json
import threading
from pathlib import Path

from .types import Trajectory


class JsonlTrajectoryWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, trajectory: Trajectory) -> None:
        payload = json.dumps(trajectory.to_dict(), ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(payload + "\n")


def load_trajectories(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows
