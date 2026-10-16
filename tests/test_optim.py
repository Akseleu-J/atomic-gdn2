import jax
import jax.numpy as jnp
import numpy as np

from atomic_gdn2_model.config import TrainConfig
from atomic_gdn2_model.optim import is_muon_param, lr_schedule, make_optimizer, newton_schulz


def _classify(tree):
    flat = jax.tree_util.tree_flatten_with_path(tree)[0]
    from atomic_gdn2_model.optim import path_to_str
    return {path_to_str(p): is_muon_param(p, x) for p, x in flat}


def test_muon_routing():
    tree = {"Embed_0": {"embedding": jnp.zeros((10, 4))}, "Block": {"Dense_0": {"kernel": jnp.zeros((4, 4)), "bias": jnp.zeros(4)}},
            "LayerNorm_0": {"scale": jnp.zeros(4)}}
    r = _classify(tree)
    assert r["embed_0/embedding"] is False
    assert r["block/dense_0/kernel"] is True
    assert r["block/dense_0/bias"] is False and r["layernorm_0/scale"] is False


def test_newton_schulz_orthogonalises():
    X = jnp.asarray(np.random.default_rng(0).normal(size=(64, 32)), jnp.float32)
    s = np.linalg.svd(np.asarray(newton_schulz(X / jnp.linalg.norm(X))), compute_uv=False)
    assert s.min() > 0.3 and s.max() < 1.3      # singular values pushed towards 1


def test_lr_schedule_shape():
    s = lr_schedule(10_000, 3e-4, 0.01, 0.1)
    assert float(s(0)) == 0.0
    assert abs(float(s(100)) - 3e-4) < 1e-9
    assert abs(float(s(10_000)) - 3e-5) < 1e-9


def test_optimizer_step_updates_all_param_kinds():
    params = {"Embed_0": {"embedding": jnp.ones((8, 4))}, "Dense_0": {"kernel": jnp.ones((4, 4)), "bias": jnp.ones(4)}}
    tx, _ = make_optimizer(1000, TrainConfig())
    state = tx.init(params)
    # step index 150 is past warmup so lr > 0
    grads = jax.tree_util.tree_map(lambda x: jnp.full_like(x, 0.1), params)
    for _ in range(150):
        upd, state = tx.update(grads, state, params)
    new = jax.tree_util.tree_map(lambda p, u: p + u, params, upd)
    for a, b in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(new)):
        assert not np.allclose(a, b) and np.isfinite(np.asarray(b)).all()
