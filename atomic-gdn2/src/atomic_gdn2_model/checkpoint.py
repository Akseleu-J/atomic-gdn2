"""Orbax checkpointing of ``{"params", "opt_state"}`` with resume + re-replication across devices."""
from __future__ import annotations

import shutil
from pathlib import Path

import jax
import orbax.checkpoint as ocp


class Checkpointer:
    def __init__(self, directory: str | Path, keep: int = 2):
        self.dir = Path(directory).resolve()
        self.mngr = ocp.CheckpointManager(str(self.dir), options=ocp.CheckpointManagerOptions(max_to_keep=keep, create=True))

    def seed_from(self, src: str | Path) -> None:
        """Copy a previous run's checkpoint directory (e.g. a read-only Kaggle input) into ours."""
        src = Path(src)
        if not src.is_dir():
            raise FileNotFoundError(f"resume source {src} does not exist")
        steps = sorted(int(d.name) for d in src.iterdir() if d.name.isdigit() and d.is_dir())
        print(f"[ckpt] seeding from {src}: steps={steps}", flush=True)
        for item in src.iterdir():
            dst = self.dir / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)
        self.mngr.reload()

    def latest_step(self):
        return self.mngr.latest_step()

    def restore_latest(self, params, opt_state, sharding=None):
        """Restore into the structure of (params, opt_state). Returns ``(params, opt_state, step)`` or None."""
        step = self.mngr.latest_step()
        if step is None:
            return None
        raw = self.mngr.restore(step, args=ocp.args.StandardRestore({"params": params, "opt_state": opt_state}))
        p, o = raw["params"], raw["opt_state"]
        if sharding is not None:   # restored arrays may come back sharded; the step functions expect replicated
            p = jax.tree_util.tree_map(lambda x: jax.device_put(x, sharding), p)
            o = jax.tree_util.tree_map(lambda x: jax.device_put(x, sharding), o)
        jax.block_until_ready((p, o))
        return p, o, int(step)

    def save(self, step: int, params, opt_state) -> bool:
        if self.mngr.latest_step() == step:      # already written (e.g. periodic save == epoch-end save)
            return False
        try:
            accepted = self.mngr.save(step, args=ocp.args.StandardSave({"params": params, "opt_state": opt_state}))
            if not accepted:
                raise RuntimeError(f"Checkpoint manager rejected step {step}")
            self.mngr.wait_until_finished()
            return True
        except Exception as e:  # noqa: BLE001 - a failed checkpoint must remain a visible run failure
            print(f"[ckpt] save failed at step {step}: {type(e).__name__}: {e}", flush=True)
            raise
