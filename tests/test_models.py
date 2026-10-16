import jax
import jax.numpy as jnp
import numpy as np
import pytest

from atomic_gdn2_model.config import ExperimentConfig
from atomic_gdn2_model.models import build_model, count_params
from atomic_gdn2_model.optim import path_to_str

pytestmark = pytest.mark.kernel
SMOKE = "configs/smoke_cpu.yaml"


def _setup(arch):
    cfg = ExperimentConfig.from_yaml(SMOKE).with_overrides([f"arch={arch}"])
    ids = jnp.asarray(np.random.default_rng(0).integers(0, cfg.model.vocab_size, (2, 128)), jnp.int32)
    model = build_model(arch, cfg.model, False)
    return cfg, ids, model, model.init(jax.random.PRNGKey(0), ids)["params"]


@pytest.mark.parametrize("arch", ["baseline", "hybrid"])
def test_forward_backward_finite(arch):
    cfg, ids, model, params = _setup(arch)
    logits = model.apply({"params": params}, ids, rngs={"dropout": jax.random.PRNGKey(1)})
    assert logits.shape == (2, 128, cfg.model.vocab_size) and bool(jnp.isfinite(logits).all())
    g = jax.grad(lambda p: jnp.mean(model.apply({"params": p}, ids, rngs={"dropout": jax.random.PRNGKey(1)}) ** 2))(params)
    assert all(bool(jnp.isfinite(x).all()) for x in jax.tree_util.tree_leaves(g))


def test_causality():
    """Changing a future token must not change earlier logits."""
    for arch in ("baseline", "hybrid"):
        cfg, ids, _, params = _setup(arch)
        ev = build_model(arch, cfg.model, True)
        a = ev.apply({"params": params}, ids)
        ids2 = ids.at[:, 100].set((ids[:, 100] + 1) % cfg.model.vocab_size)
        b = ev.apply({"params": params}, ids2)
        np.testing.assert_allclose(np.asarray(a[:, :100]), np.asarray(b[:, :100]), atol=1e-5)
        assert not np.allclose(np.asarray(a[:, 100:]), np.asarray(b[:, 100:]))


def test_param_names_are_checkpoint_compatible():
    """Flax param paths are the on-disk checkpoint schema; renaming classes breaks old checkpoints."""
    _, _, _, pb = _setup("baseline")
    names = {path_to_str(p).split("/")[0] for p, _ in jax.tree_util.tree_flatten_with_path(pb)[0]}
    assert "checkpointbaselineblock_0" in names and "embed_0" in names and "layernorm_0" in names
    _, _, _, ph = _setup("hybrid")
    paths = {path_to_str(p) for p, _ in jax.tree_util.tree_flatten_with_path(ph)[0]}
    for must in ("block_0/layer_1/dar/query", "block_0/layer_0/sublayer/dense_0/kernel",
                 "block_0/ffn/dense_0/kernel", "block_0/layer_0/pre_sublayer_norm/scale",
                 "block_0/ffn_dar/key_norm/scale"):
        assert must in paths, must


def test_history_dar_receives_task_gradient():
    """DAR retrieval feeds the sublayer input; learned routing must receive task gradients."""
    cfg, ids, model, params = _setup("hybrid")
    g = jax.grad(lambda p: jnp.mean(model.apply({"params": p}, ids, rngs={"dropout": jax.random.PRNGKey(1)}) ** 2))(params)
    flat = jax.tree_util.tree_flatten_with_path(g)[0]
    dar = [x for p, x in flat if "/dar/" in path_to_str(p)]
    intra = [x for p, x in flat if "/intra/" in path_to_str(p)]
    assert dar and all(bool(jnp.isfinite(x).all()) for x in dar)
    assert max(float(jnp.abs(x).max()) for x in dar) > 0.0
    assert not intra, 'The old standalone intra-block Q/K router is not in the current model'
    assert count_params(params) > 0
