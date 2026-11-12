"""Multi-head Latent Attention sublayer (DeepSeek-style low-rank KV, decoupled RoPE)."""
from __future__ import annotations

import math

import jax.numpy as jnp
from flax import linen as nn

from ..config import ModelConfig
from .common import RoPE, apply_rope, flash_or_softmax


class MLASub(nn.Module):
    cfg: ModelConfig
    deterministic: bool = False

    @nn.compact
    def __call__(self, x):
        c = self.cfg
        B, L, _ = x.shape
        H = c.n_heads
        d_nope, d_rope = c.qk_nope_head_dim, c.qk_rope_head_dim
        d_head = d_nope + d_rope
        cd, pd = c.cdtype, c.pdtype

        if c.q_lora_rank is None:
            q = nn.Dense(H * d_head, use_bias=False, dtype=cd, param_dtype=pd)(x)
        else:
            c_Q = nn.Dense(c.q_lora_rank, use_bias=False, dtype=cd, param_dtype=pd)(x)
            q = nn.Dense(H * d_head, use_bias=False, dtype=cd, param_dtype=pd)(c_Q)
        q = q.reshape(B, L, H, d_head)

        c_KV = nn.Dense(c.kv_lora_rank, use_bias=False, dtype=cd, param_dtype=pd)(x)
        c_KV = nn.LayerNorm(dtype=cd, param_dtype=pd)(c_KV)
        k_nope = nn.Dense(H * d_nope, use_bias=False, dtype=cd, param_dtype=pd)(c_KV).reshape(B, L, H, d_nope)
        v = nn.Dense(H * d_head, use_bias=False, dtype=cd, param_dtype=pd)(c_KV).reshape(B, L, H, d_head)
        k_rope = nn.Dense(d_rope, use_bias=False, dtype=cd, param_dtype=pd)(x).reshape(B, L, 1, d_rope)

        cos, sin = RoPE(dim=d_rope, theta=c.rope_theta)(L)
        q_rope, q_nope = q[..., d_nope:], q[..., :d_nope]
        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin)

        q_full = jnp.concatenate([q_nope, q_rope], axis=-1)
        k_full = jnp.concatenate([k_nope, jnp.broadcast_to(k_rope, (B, L, H, d_rope))], axis=-1)

        o = flash_or_softmax(q_full, k_full, v, 1.0 / math.sqrt(d_head), causal=True,
                             interpret=c.kernel_interpret)
        out = nn.Dense(c.d_model, use_bias=False, dtype=cd, param_dtype=pd)(o.reshape(B, L, H * d_head))
        return nn.Dropout(rate=c.dropout_rate, deterministic=self.deterministic)(out)
