"""Paper Eq. 1-2 routing on the existing GDN2/MLA backbone and block hierarchy.

Production hybrid: promoted from the measured paper_dar experiment.
Sources: https://arxiv.org/html/2605.18855v1#S2.SS3
Keep the same token-mixing/FFN layout, block boundaries, residual sanitization,
embedding/head and rematerialization; change the residual routing formulation.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn
from ..config import ModelConfig
from .common import SwiGLU
from .gdn2 import GDN2Sub
from .mla import MLASub
def _sanitize(x, lim=1e3):
    return jnp.nan_to_num(jnp.clip(x, -lim, lim), nan=0.0, posinf=lim, neginf=-lim)


class PaperDAR(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, residual, sources):
        if not sources:
            return residual
        values = jnp.stack(sources)
        keys = nn.RMSNorm(epsilon=1e-6, dtype=self.cfg.cdtype,
                          param_dtype=self.cfg.pdtype, name='key_norm')(values)
        query = self.param('query', nn.initializers.zeros, (self.cfg.d_model,), self.cfg.pdtype)
        scores = jnp.einsum('d,nbtd->nbt', query.astype(jnp.float32), keys.astype(jnp.float32))
        weights = jax.nn.softmax(scores, axis=0)
        selected = jnp.einsum('nbt,nbtd->btd', weights, values.astype(jnp.float32))
        return residual + selected.astype(residual.dtype)


class PaperLayer(nn.Module):
    cfg: ModelConfig
    layer_type: str
    deterministic: bool

    @nn.compact
    def __call__(self, x, local, history):
        sources = [history[j] for j in range(history.shape[0])] + list(local)
        routed = PaperDAR(self.cfg, name='dar')(x, sources)
        routed = nn.RMSNorm(epsilon=1e-6, name='pre_sublayer_norm')(routed)
        sub = GDN2Sub if self.layer_type == 'gdn2' else MLASub
        return _sanitize(sub(self.cfg, self.deterministic, name='sublayer')(routed))


class PaperBlock(nn.Module):
    cfg: ModelConfig
    block_idx: int
    deterministic: bool

    @nn.compact
    def __call__(self, x, history):
        start_x, local = x, []
        for i in range(self.cfg.layers_per_block):
            lt = self.cfg.layer_types[self.block_idx * self.cfg.layers_per_block + i]
            delta = PaperLayer(self.cfg, lt, self.deterministic, name=f'layer_{i}')(x, local, history)
            local.append(delta)
            x = _sanitize(x + delta)
        sources = [history[j] for j in range(history.shape[0])] + local
        routed = PaperDAR(self.cfg, name='ffn_dar')(x, sources)
        h = nn.LayerNorm(dtype=self.cfg.cdtype, param_dtype=self.cfg.pdtype)(routed)
        x = x + SwiGLU(self.cfg, self.deterministic, name='ffn')(h)
        # Actual block change includes the FFN; grouping boundaries stay identical.
        return x, jnp.concatenate([history, (x - start_x)[None]], axis=0)


class LM(nn.Module):
    cfg: ModelConfig
    deterministic: bool = False

    @nn.compact
    def __call__(self, ids):
        c = self.cfg
        emb = nn.Embed(c.vocab_size, c.d_model, embedding_init=nn.initializers.normal(.02),
                       dtype=c.cdtype, param_dtype=c.pdtype)
        x = emb(ids)
        history = jnp.zeros((0,) + x.shape, x.dtype)
        for i in range(c.num_blocks):
            x, history = nn.remat(PaperBlock)(c, i, self.deterministic, name=f'block_{i}')(x, history)
        x = nn.LayerNorm(dtype=c.cdtype, param_dtype=c.pdtype)(x)
        if c.tie_embeddings:
            return jnp.dot(x.astype(jnp.float32), emb.embedding.astype(jnp.float32).T)
        return nn.Dense(c.vocab_size, use_bias=False, kernel_init=nn.initializers.normal(.02),
                        dtype=jnp.float32, param_dtype=c.pdtype)(x.astype(jnp.float32))
