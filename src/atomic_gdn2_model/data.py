"""OpenWebText (GPT-2 BPE, uint16 ``.bin`` files) token loading and batching."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np

from .config import DataConfig


def find_data_dir(cfg: DataConfig) -> Path:
    cands = (cfg.dir,) if cfg.dir else cfg.candidates
    for d in cands:
        if d and os.path.isfile(os.path.join(d, cfg.train_file)) and os.path.isfile(os.path.join(d, cfg.val_file)):
            return Path(d)
    raise FileNotFoundError(
        f"{cfg.train_file}/{cfg.val_file} not found in {list(cands)}. On Kaggle add the dataset "
        f"'windmaple/openwebtext-gpt2'; elsewhere set data.dir (see docs/REPRODUCING.md).")


def load_tokens(cfg: DataConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (train, val) uint16 arrays truncated to ``train_tokens`` / ``val_tokens``."""
    d = find_data_dir(cfg)
    tr = np.memmap(d / cfg.train_file, dtype=np.uint16, mode="r")
    va = np.memmap(d / cfg.val_file, dtype=np.uint16, mode="r")
    print(f"[data] {d}  train={len(tr):,}  val={len(va):,} tokens on disk", flush=True)
    tr, va = tr[: cfg.train_tokens], va[: cfg.val_tokens]
    if cfg.in_memory:
        tr, va = np.array(tr, dtype=np.uint16), np.array(va, dtype=np.uint16)
    return tr, va


class Loader:
    """Non-overlapping ``(seq_len + 1)`` windows -> ``{"input_ids", "labels"}`` int32 batches.

    ``epoch(ep)`` uses ``default_rng(seed + ep)``, so epoch ``ep`` has the same order in a fresh run
    and in a run resumed at that epoch; ``skip`` fast-forwards inside an epoch (mid-epoch resume).
    """

    def __init__(self, tokens, seq_len: int, batch_size: int, sharding=None, shuffle: bool = True, seed: int = 42):
        self.tokens, self.sl, self.bsz = tokens, seq_len, batch_size
        self.sharding, self.shuffle, self.seed = sharding, shuffle, seed
        n_windows = (len(tokens) - 1) // (seq_len + 1)
        self.steps_per_epoch = n_windows // batch_size
        self.n_windows = self.steps_per_epoch * batch_size
        if self.steps_per_epoch == 0:
            raise ValueError(f"not enough tokens ({len(tokens)}) for one batch of {batch_size}x{seq_len}")

    def _batch(self, idx) -> dict:
        w = np.stack([self.tokens[i * (self.sl + 1): (i + 1) * (self.sl + 1)] for i in idx]).astype(np.int32)
        x, y = jnp.asarray(w[:, :-1]), jnp.asarray(w[:, 1:])
        if self.sharding is not None:
            x, y = jax.device_put(x, self.sharding), jax.device_put(y, self.sharding)
        return {"input_ids": x, "labels": y}

    def epoch(self, epoch: int = 0, skip: int = 0) -> Iterator[dict]:
        rng = np.random.default_rng(self.seed + epoch)
        perm = rng.permutation(self.n_windows) if self.shuffle else np.arange(self.n_windows)
        for s in range(skip, self.steps_per_epoch):
            yield self._batch(perm[s * self.bsz:(s + 1) * self.bsz])

    def eval_epoch(self, max_batches: int | None = None) -> Iterator[dict]:
        n = self.steps_per_epoch if max_batches is None else min(max_batches, self.steps_per_epoch)
        for s in range(n):
            yield self._batch(np.arange(s * self.bsz, (s + 1) * self.bsz))
