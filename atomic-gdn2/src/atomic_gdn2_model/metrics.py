"""Append-only JSONL metrics log (one record per event) next to the run's checkpoints."""
from __future__ import annotations

import json
import time
from pathlib import Path


class MetricsLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, kind: str, **fields) -> None:
        rec = {"kind": kind, "time": time.time(), **fields}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
