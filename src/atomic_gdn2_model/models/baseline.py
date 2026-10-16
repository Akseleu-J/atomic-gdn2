"""Pure GDN-2 baseline: `layers_per_block` x (LN -> GDN-2 -> residual), then one SwiGLU, per block."""
from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from ..config import ModelConfig
from .common import SwiGLU
from .gdn2 import GDN2Sub


class BaselineBlock(nn.Module):
    cfg: ModelConfig
    gdn2_per_block: int
    deterministic: bool = False

    @nn.compact
    def __call__(self, x):
        cd, pd = self.cfg.cdtype, self.cfg.pdtype
        for _ in range(self.gdn2_per_block):
            h = nn.LayerNorm(dtype=cd, param_dtype=pd)(x)
            x = x + GDN2Sub(self.cfg, self.deterministic)(h)
        h = nn.LayerNorm(dtype=cd, param_dtype=pd)(x)
        return x + SwiGLU(self.cfg, self.deterministic)(h)


class BaselineLM(nn.Module):
    cfg: ModelConfig
    n_blocks: int
    gdn2_per_block: int
    deterministic: bool = False

    @nn.compact
    def __call__(self, ids):
        c = self.cfg
        cd, pd = c.cdtype, c.pdtype
        emb = nn.Embed(c.vocab_size, c.d_model, embedding_init=nn.initializers.normal(0.02),
                       dtype=cd, param_dtype=pd)
        x = emb(ids)
        for _ in range(self.n_blocks):
            x = nn.remat(BaselineBlock)(c, self.gdn2_per_block, self.deterministic)(x)
        x = nn.LayerNorm(dtype=cd, param_dtype=pd)(x)
        x32 = x.astype(jnp.float32)
        if c.tie_embeddings:
            return jnp.dot(x32, emb.embedding.astype(jnp.float32).T)
        return nn.Dense(c.vocab_size, use_bias=False, kernel_init=nn.initializers.normal(0.02),
                        dtype=jnp.float32, param_dtype=pd)(x32)
