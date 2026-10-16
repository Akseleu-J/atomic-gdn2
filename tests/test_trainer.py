import json

import numpy as np
import pytest

from atomic_gdn2_model.config import ExperimentConfig
from atomic_gdn2_model.trainer import run

pytestmark = pytest.mark.kernel


def _records(run_dir):
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("arch", ["baseline", "hybrid"])
def test_train_then_resume(tmp_path, arch):
    rng = np.random.default_rng(0)
    tokens = (rng.integers(0, 256, 20000).astype(np.uint16), rng.integers(0, 256, 5000).astype(np.uint16))
    cfg = ExperimentConfig.from_yaml("configs/smoke_cpu.yaml").with_overrides([f"arch={arch}", "train.max_steps=2"])
    s1 = run(cfg, tmp_path / "a", tokens=tokens)
    assert s1["final_step"] == 2 and s1["skipped_steps"] == 0 and np.isfinite(s1["best_val_bpb_est"])
    assert (tmp_path / "a" / "config.yaml").exists()

    # continue from a copy of the checkpoints in a new run dir
    cfg2 = cfg.with_overrides(["train.max_steps=4"])
    s2 = run(cfg2, tmp_path / "b", resume_from=tmp_path / "a" / "checkpoints", tokens=tokens)
    assert s2["final_step"] == 4
    recs = _records(tmp_path / "b")
    assert recs[0]["kind"] == "start" and recs[0]["step"] == 2
    steps = [r["step"] for r in recs if r["kind"] == "train"]
    assert steps == [3, 4]
    assert all(np.isfinite(r["loss"]) for r in recs if r["kind"] == "train")
