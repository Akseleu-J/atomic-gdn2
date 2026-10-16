"""Optimiser: Muon (Newton-Schulz orthogonalised momentum) for 2-D hidden matrices, AdamW elsewhere,
global-norm clipping and a warmup + cosine schedule applied to both."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from .config import TrainConfig

_NON_MUON = ("embed", "embedding", "head", "lm_head")


def path_to_str(path) -> str:
    parts = []
    for p in path:
        for a in ("key", "name", "idx"):
            if hasattr(p, a):
                parts.append(str(getattr(p, a)))
                break
        else:
            parts.append(str(p))
    return "/".join(parts).lower()


def is_muon_param(path, p) -> bool:
    """Muon: 2-D matrices that are not embeddings / output heads."""
    if p.ndim != 2:
        return False
    name = path_to_str(path)
    return not any(x in name for x in _NON_MUON)


def newton_schulz(X, steps: int = 5):
    a, b, c = 3.4445, -4.7750, 2.0315
    tall = X.shape[-2] > X.shape[-1]
    if tall:
        X = jnp.swapaxes(X, -2, -1)
    X = X / (jnp.linalg.norm(X, axis=(-2, -1), keepdims=True) * 1.01 + 1e-7)
    for _ in range(steps):
        A = X @ jnp.swapaxes(X, -2, -1)
        X = a * X + (b * A + c * (A @ A)) @ X
    if tall:
        X = jnp.swapaxes(X, -2, -1)
    return X


def _orthogonalize(g, ns: int = 5):
    n = jnp.linalg.norm(g)
    X = g / jnp.where(n < 1e-7, 1.0, n)
    return jnp.nan_to_num(newton_schulz(X, ns), nan=0.0)


def muon_tx(momentum: float = 0.95, ns: int = 5, wd: float = 0.02) -> optax.GradientTransformation:
    """Returns *negative* updates (descent direction, shape-scaled) incl. decoupled weight decay;
    the learning rate is applied afterwards by ``optax.scale_by_schedule``."""

    def init(params):
        return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x, jnp.float32), params)

    def update(grads, mu, params=None):
        def f(g, m, p):
            combined = g.astype(jnp.float32) + momentum * m.astype(jnp.float32)
            X = _orthogonalize(combined, ns)
            scale = 0.2 * jnp.sqrt(float(max(p.shape)))
            return -(scale * X + wd * p.astype(jnp.float32)).astype(g.dtype)

        upd = jax.tree_util.tree_map(f, grads, mu, params)
        new_mu = jax.tree_util.tree_map(
            lambda m, g: (momentum * m.astype(jnp.float32) + g.astype(jnp.float32)).astype(m.dtype), mu, grads)
        return upd, new_mu

    return optax.GradientTransformation(init, update)


def lr_schedule(total_steps: int, peak: float, warmup_ratio: float, min_ratio: float):
    warm = max(100, int(total_steps * warmup_ratio))
    return optax.join_schedules(
        [optax.linear_schedule(0.0, peak, warm),
         optax.cosine_decay_schedule(peak, max(1, total_steps - warm), alpha=min_ratio)], [warm])


def make_optimizer(total_steps: int, t: TrainConfig, *, external_lr: bool = False):
    """Returns ``(tx, schedule)``. ``total_steps`` = number of optimizer steps in the whole run."""
    sched = lr_schedule(total_steps, t.peak_lr, t.warmup_ratio, t.min_lr_ratio)
    m_mask = lambda p: jax.tree_util.tree_map_with_path(lambda path, x: is_muon_param(path, x), p)
    a_mask = lambda p: jax.tree_util.tree_map_with_path(lambda path, x: not is_muon_param(path, x), p)
    tx = optax.chain(
        optax.clip_by_global_norm(t.grad_clip),
        optax.masked(muon_tx(t.muon_momentum, t.muon_ns_steps, t.weight_decay), m_mask),
        optax.masked(optax.adamw(learning_rate=1.0, weight_decay=t.weight_decay, b1=t.adam_b1, b2=t.adam_b2), a_mask),
        optax.scale_by_schedule(optax.constant_schedule(1.0) if external_lr else sched),
    )
    return tx, sched
