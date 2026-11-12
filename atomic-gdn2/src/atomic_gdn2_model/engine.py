"""Loss, jitted train/eval steps (gradient accumulation + non-finite-gradient skip), evaluation."""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import optax


def make_loss_fns(model_train, model_eval):
    def _ce(logits, labels):
        lp = jax.nn.log_softmax(logits.astype(jnp.float32), -1)
        return -jnp.mean(jnp.take_along_axis(lp, labels[..., None], -1))

    def loss_train(p, b, rng):
        return _ce(model_train.apply({"params": p}, b["input_ids"], rngs={"dropout": rng}), b["labels"])

    def loss_eval(p, b):
        return _ce(model_eval.apply({"params": p}, b["input_ids"]), b["labels"])

    return loss_train, loss_eval


def build_steps(loss_train, tx, param_sharding, accum: int, *, external_lr: bool = False):
    """Returns ``(micro, apply)`` jitted functions.

    ``micro(params, grad_acc, batch, rng) -> (grad_acc, loss, micro_grad_norm)``
    ``apply(params, opt_state, grad_acc) -> (params, opt_state, finite, grad_norm)``; the update is
    skipped (params/opt_state unchanged) when the averaged gradient norm is not finite.
    """
    vg = jax.value_and_grad(loss_train, argnums=0)

    def micro(p, acc, b, rng):
        l, g = vg(p, b, rng)
        g = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, g)
        return jax.tree_util.tree_map(lambda x, y: x + y, acc, g), l, optax.global_norm(g)

    def apply_scaled(p, o, acc, lr_scale):
        avg = jax.tree_util.tree_map(lambda g: g / accum, acc)
        gn = optax.global_norm(avg)
        fin = jnp.isfinite(gn)

        def do(_):
            u, s = tx.update(avg, o, p)
            u = jax.tree_util.tree_map(lambda update: update * lr_scale, u)
            return optax.apply_updates(p, u), s

        new_p, new_o = jax.lax.cond(fin, do, lambda _: (p, o), None)
        return new_p, new_o, fin, gn

    apply = apply_scaled if external_lr else lambda p, o, acc: apply_scaled(p, o, acc, 1.0)

    if param_sharding is None:
        return jax.jit(micro, donate_argnums=(1,)), jax.jit(apply, donate_argnums=(0, 1, 2))
    return (jax.jit(micro, donate_argnums=(1,), in_shardings=(param_sharding, param_sharding, None, None),
                    out_shardings=(param_sharding, None, None)),
            jax.jit(apply, donate_argnums=(0, 1, 2), in_shardings=(param_sharding, None, param_sharding) + ((None,) if external_lr else ()),
                    out_shardings=(param_sharding, None, None, None)))


def to_bpb(loss: float, bytes_per_token: float) -> float:
    """Estimated bits-per-byte from a token-level cross-entropy (nats)."""
    return loss / math.log(2.0) / bytes_per_token


def evaluate(eval_jit, params, loader, max_batches, bytes_per_token: float):
    tot, n = 0.0, 0
    for b in loader.eval_epoch(max_batches):
        value = float(jax.device_get(eval_jit(params, b)))
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite validation loss")
        tot += value
        n += 1
    if n == 0:
        raise ValueError("Validation loader yielded no batches")
    mean = tot / n
    return mean, to_bpb(mean, bytes_per_token)
