import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from atomic_gdn2_model.engine import build_steps
from atomic_gdn2_model.optim import make_optimizer
from atomic_gdn2_model.config import TrainConfig


@pytest.mark.parametrize('lr', [0., .1])
def test_external_lr_scales_real_optimizer_update(lr):
    loss = lambda p, b, r: jnp.sum(p['w'] ** 2)
    tx = optax.sgd(1.)
    _, apply = build_steps(loss, tx, None, 1, external_lr=True)
    params = {'w': jnp.array([1., 2.])}
    state = tx.init(params)
    grads = {'w': jnp.array([2., 4.])}
    new_params, _, finite, _ = apply(params, state, grads, lr)
    np.testing.assert_allclose(np.asarray(new_params['w']), [1 - 2 * lr, 2 - 4 * lr], rtol=1e-6)
    assert bool(finite)


def test_default_engine_keeps_three_argument_interface():
    tx = optax.sgd(.1)
    _, apply = build_steps(lambda p, b, r: jnp.sum(p['w'] ** 2), tx, None, 1)
    p = {'w': jnp.array([1., 2.])}
    new, _, finite, _ = apply(p, tx.init(p), {'w': jnp.array([2., 4.])})
    np.testing.assert_allclose(np.asarray(new['w']), [.8, 1.6], rtol=1e-6)
    assert bool(finite)


def test_clock_optimizer_does_not_apply_a_second_step_schedule():
    tx, _ = make_optimizer(1000, TrainConfig(), external_lr=True)
    p = {'Embed_0': {'embedding': jnp.ones((8, 4))}}
    g = jax.tree_util.tree_map(lambda v: jnp.full_like(v, .1), p)
    updates, _ = tx.update(g, tx.init(p), p)
    assert np.max(np.abs(np.asarray(updates['Embed_0']['embedding']))) > 0
