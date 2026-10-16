"""GDN-2 token-mixing sublayer (projections + atomic-ops BLR kernel)."""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..config import ModelConfig
from ..distributed import shard_map_auto
from ..kernels import get_blr


class GDN2Sub(nn.Module):
    cfg: ModelConfig
    deterministic: bool = False

    @nn.compact
    def __call__(self, x):
        c = self.cfg
        B, L, _ = x.shape
        H, Dh = c.n_heads, c.d_head
        HD = H * Dh
        cd, pd = c.cdtype, c.pdtype

        q = nn.Dense(HD, use_bias=False, dtype=cd, param_dtype=pd)(x).reshape(B, L, H, Dh)
        k = nn.Dense(HD, use_bias=False, dtype=cd, param_dtype=pd)(x).reshape(B, L, H, Dh)
        v = nn.Dense(HD, use_bias=False, dtype=cd, param_dtype=pd)(x).reshape(B, L, H, Dh)
        w = nn.Dense(HD, use_bias=False, dtype=cd, param_dtype=pd)(x).reshape(B, L, H, Dh)
        beta = nn.Dense(HD, use_bias=False, dtype=cd, param_dtype=pd)(x).reshape(B, L, H, Dh)
        gb = nn.initializers.constant(math.log(math.expm1(c.gdn2_g_init)))
        gr = nn.Dense(HD, use_bias=True, bias_init=gb, dtype=cd, param_dtype=pd)(x).reshape(B, L, H, Dh)

        if c.l2norm_qk:
            q = q * jax.lax.rsqrt(jnp.sum(q * q, -1, keepdims=True) + 1e-6)
            k = k * jax.lax.rsqrt(jnp.sum(k * k, -1, keepdims=True) + 1e-6)

        blr, kcfg = get_blr(c.gdn2_chunk_size, c.gdn2_bs2, c.gdn2_mb, 1.0 / math.sqrt(Dh), c.kernel_interpret)
        b = jax.nn.sigmoid(beta)
        g = -jnp.minimum(jax.nn.softplus(gr), kcfg.g_max)   # domain: g in [-g_max, 0]

        to_hm = lambda t: t.transpose(0, 2, 1, 3).astype(jnp.float32)
        q32, k32, v32, w32, b32, g32 = map(to_hm, (q, k, v, w, b, g))

        def inner(q_, k_, v_, w_, b_, g_):
            o_, _ = blr(q_, k_, v_, w_, b_, g_, h0=None)
            return o_

        o = shard_map_auto(inner, 6)(q32, k32, v32, w32, b32, g32)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, HD).astype(cd)
        out = nn.Dense(c.d_model, use_bias=False, dtype=cd, param_dtype=pd)(o)
        return nn.Dropout(rate=c.dropout_rate, deterministic=self.deterministic)(out)
