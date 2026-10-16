"""Exercise real JAX train steps and timed finalization; checkpoint I/O is a recording fake."""
import importlib
import sys
import types
import numpy as np
from flax import linen as nn
from atomic_gdn2_model.config import ExperimentConfig


def test_timed_training_finalizes_after_steps(monkeypatch, tmp_path):
    saved = []
    closed = []
    class Checkpointer:
        def __init__(self, *args, **kwargs):
            self.mngr = types.SimpleNamespace(close=lambda: closed.append(True))
        def restore_latest(self, *args):
            return None
        def save(self, step, *args):
            saved.append(step)
            return True
    fake = types.ModuleType('atomic_gdn2_model.checkpoint')
    fake.Checkpointer = Checkpointer
    monkeypatch.setitem(sys.modules, 'atomic_gdn2_model.checkpoint', fake)
    trainer = importlib.import_module('atomic_gdn2_model.trainer')
    monkeypatch.setattr(trainer, 'Checkpointer', Checkpointer)
    class Budget:
        checks = 0
        def expired(self):
            self.checks += 1
            return self.checks > 2
        def remaining(self):
            return 100.
        def learning_rate(self, *args):
            return .01
    monkeypatch.setattr(trainer.WallClockBudget, 'from_deadline', lambda *args: Budget())
    class TinyLM(nn.Module):
        @nn.compact
        def __call__(self, ids):
            return nn.Dense(32)(nn.Embed(32, 8)(ids))
    monkeypatch.setattr(trainer, 'build_model', lambda *args: TinyLM())
    cfg = ExperimentConfig.from_dict(dict(model=dict(vocab_size=32, d_model=8, n_heads=1, d_ff=16,
        dropout_rate=0., use_bf16=False), train=dict(seq_len=128, batch_size=2, epochs=1, log_every=1,
        eval_every=0, ckpt_every=0), runtime=dict(shard=False, require_tpu=False, wall_deadline_unix=1e15)))
    rng = np.random.default_rng(42)
    tokens = (rng.integers(0, 32, 5000).astype(np.uint16), rng.integers(0, 32, 1000).astype(np.uint16))
    summary = trainer.run(cfg, tmp_path / 'timed', tokens=tokens)
    assert summary['final_step'] == 2 and summary['stop_reason'] == 'wall_time_budget'
    assert summary['tokens_seen'] == 512 and not summary['completed_data_pass']
    assert summary['skipped_steps'] == 0 and np.isfinite(summary['best_val_bpb_est'])
    assert saved == [2] and closed == [True]
