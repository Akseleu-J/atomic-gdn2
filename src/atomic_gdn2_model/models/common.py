"""Shared building blocks: RoPE, flash/softmax attention, SwiGLU.

NOTE: Flax derives parameter names from class names and creation order. Do NOT rename classes or
reorder layer construction in this package - existing orbax checkpoints depend on it.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..config import ModelConfig
from ..distributed import shard_map_auto

try:
    from jax.experimental.pallas.ops.tpu.flash_attention import (
        BlockSizes as _FlashBlockSizes,
    )
    from jax.experimental.pallas.ops.tpu.flash_attention import (
        flash_attention as _pallas_flash_attention,
    )
    HAS_FLASH = True
except Exception:  # pragma: no cover - non-TPU builds
    _pallas_flash_attention = None
    _FlashBlockSizes = None
    HAS_FLASH = False


def flash_or_softmax(q, k, v, scale, causal=True, *, interpret=False):
    """(B, L, H, D) causal attention: Pallas flash kernel on TPU, dense fp32 softmax elsewhere."""
    if HAS_FLASH and jax.default_backend() == "tpu" and not interpret:
        L = q.shape[1]
        s = min(512, L)
        bs = _FlashBlockSizes(
            block_q=s, block_k_major=s, block_k=s, block_b=1,
            block_q_major_dkv=s, block_k_major_dkv=s, block_k_dkv=s, block_q_dkv=s,
            block_k_major_dq=s, block_k_dq=s, block_q_dq=s)

        def inner(q_, k_, v_):
            return _pallas_flash_attention(q_, k_, v_, block_sizes=bs, causal=causal, sm_scale=float(scale))

        qt, kt, vt = (jnp.swapaxes(t, 1, 2) for t in (q, k, v))
        return jnp.swapaxes(shard_map_auto(inner, 3)(qt, kt, vt), 1, 2)
    attn = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) * scale
    if causal:
        m = jnp.tril(jnp.ones((q.shape[1], k.shape[1])))
        attn = jnp.where(m, attn, -1e9)
    attn = jax.nn.softmax(attn, -1).astype(q.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", attn, v)


class RoPE(nn.Module):
    dim: int
    theta: float = 10000.0

    @nn.compact
    def __call__(self, L):
        inv = 1.0 / (self.theta ** (jnp.arange(0, self.dim, 2) / self.dim))
        t = jnp.arange(L, dtype=jnp.float32)
        f = jnp.einsum("i,j->ij", t, inv)
        e = jnp.concatenate([f, f], -1)
        return jnp.cos(e), jnp.sin(e)


def apply_rope(x, c, s):
    c = c[None, :, None, :].astype(x.dtype)
    s = s[None, :, None, :].astype(x.dtype)
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * c + jnp.concatenate([-x2, x1], -1) * s


class SwiGLU(nn.Module):
    cfg: ModelConfig
    deterministic: bool = False

    @nn.compact
    def __call__(self, x):
        d, cd, pd = self.cfg.d_ff, self.cfg.cdtype, self.cfg.pdtype
        g = nn.Dense(d, use_bias=False, dtype=cd, param_dtype=pd)(x)
        u = nn.Dense(d, use_bias=False, dtype=cd, param_dtype=pd)(x)
        out = nn.Dense(self.cfg.d_model, use_bias=False, dtype=cd, param_dtype=pd)(jax.nn.silu(g) * u)
        return nn.Dropout(rate=self.cfg.dropout_rate, deterministic=self.deterministic)(out)
