"""End-to-end training driver (data-parallel, Muon+AdamW, resumable)."""
from __future__ import annotations

import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from . import distributed as dist
from .budget import WallClockBudget
from .checkpoint import Checkpointer
from .config import ExperimentConfig
from .data import Loader, load_tokens
from .engine import build_steps, evaluate, make_loss_fns, to_bpb
from .metrics import MetricsLogger
from .models import build_model, count_params
from .optim import make_optimizer


def run(cfg: ExperimentConfig, run_dir: str | Path, resume_from: str | Path | None = None, tokens=None) -> dict:
    """Train ``cfg`` and return a summary dict. ``tokens=(train, val)`` overrides data loading (tests)."""
    t, m = cfg.train, cfg.model
    budget = (WallClockBudget.from_deadline(cfg.runtime.wall_deadline_unix, cfg.runtime.checkpoint_margin_s)
              if cfg.runtime.wall_deadline_unix is not None else None)
    if budget and (resume_from or t.accum_steps != 1):
        raise ValueError('Time-budget runs require a fresh run with accum_steps=1')
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(cfg.to_yaml())
    log = MetricsLogger(run_dir / "metrics.jsonl")

    # ---- runtime / mesh -------------------------------------------------------------------
    backend, n_dev = dist.wait_for_backend(cfg.runtime.require_tpu, cfg.runtime.backend_wait_s)
    mesh = dist.build_mesh() if cfg.runtime.shard else None
    dist.set_mesh(mesh)
    if mesh is not None and t.batch_size % n_dev:
        raise ValueError(f"train.batch_size={t.batch_size} must be divisible by n_devices={n_dev}")
    rep, bsh = dist.replicated(mesh), dist.batch_sharding(mesh)
    print(f"[setup] name={cfg.name} arch={cfg.arch} backend={backend} devices={n_dev} sharded={mesh is not None}", flush=True)

    # ---- data -----------------------------------------------------------------------------
    train_tok, val_tok = tokens if tokens is not None else load_tokens(cfg.data)
    train_loader = Loader(train_tok, t.seq_len, t.batch_size, bsh, shuffle=True, seed=t.seed_data)
    val_loader = Loader(val_tok, t.seq_len, t.batch_size, bsh, shuffle=False)
    opt_spe = train_loader.steps_per_epoch // t.accum_steps          # optimizer steps / epoch
    total_steps = opt_spe * t.epochs
    stop_at = min(total_steps, t.max_steps) if t.max_steps else total_steps
    print(f"[train] steps/epoch={opt_spe} total_steps={total_steps} "
          f"tokens/step={t.batch_size * t.accum_steps * t.seq_len:,} "
          f"total_tokens={total_steps * t.batch_size * t.accum_steps * t.seq_len:,}", flush=True)

    # ---- model / params ---------------------------------------------------------------------
    model_train, model_eval = build_model(cfg.arch, m, False), build_model(cfg.arch, m, True)
    rng = jax.random.PRNGKey(t.seed_init)
    init_ids = jnp.zeros((n_dev if mesh is not None else 1, m.gdn2_chunk_size), jnp.int32)
    init_fn = lambda r: model_train.init(r, init_ids)["params"]
    if mesh is not None:
        psh = jax.tree_util.tree_map(lambda _: rep, jax.eval_shape(init_fn, rng))
        params = jax.jit(init_fn, out_shardings=psh)(rng)
    else:
        psh, params = None, jax.jit(init_fn)(rng)
    jax.block_until_ready(params)
    n_params = count_params(params)
    print(f"[model] params={n_params:,}", flush=True)

    # ---- optimiser / steps ------------------------------------------------------------------
    tx, sched = make_optimizer(total_steps, t, external_lr=budget is not None)
    ost = jax.jit(tx.init)(params)
    zero_fn = jax.jit(lambda p: jax.tree_util.tree_map(jnp.zeros_like, p), **({"out_shardings": psh} if psh else {}))
    loss_train, loss_eval = make_loss_fns(model_train, model_eval)
    micro_jit, apply_jit = build_steps(loss_train, tx, psh, t.accum_steps, external_lr=budget is not None)
    eval_jit = jax.jit(loss_eval)
    bpt = cfg.data.bytes_per_token

    # ---- checkpoint / resume ----------------------------------------------------------------
    ckpt = Checkpointer(run_dir / "checkpoints", keep=t.ckpt_keep)
    if resume_from:
        ckpt.seed_from(resume_from)
    gs = 0
    restored = ckpt.restore_latest(params, ost, rep)
    if restored is not None:
        if budget:
            raise ValueError('Time-budget runs must not restore an existing checkpoint')
        params, ost, gs = restored
        print(f"[resume] restored step {gs}", flush=True)
    else:
        print("[resume] no checkpoint, starting from scratch", flush=True)
    start_step = gs
    log.log("start", step=gs, n_params=n_params, total_steps=total_steps, arch=cfg.arch, name=cfg.name)

    # ---- loop --------------------------------------------------------------------------------
    start_epoch, skip_opt = divmod(gs, opt_spe)
    best_bpb, skipped = float("inf"), 0
    t0 = time.perf_counter()
    acc, ia, loss_sum = zero_fn(params), 0, 0.0
    drop_rng = jax.random.PRNGKey(t.seed_dropout)
    if gs:                               # keep the dropout stream distinct from the pre-resume one
        drop_rng = jax.random.fold_in(drop_rng, gs)
    done = gs >= stop_at
    stop_reason = 'max_steps' if stop_at < total_steps else 'data_complete'

    for ep in range(start_epoch, t.epochs):
        if done:
            break
        e0 = time.perf_counter()
        skip_micro = skip_opt * t.accum_steps if ep == start_epoch else 0
        for batch in train_loader.epoch(ep, skip=skip_micro):
            if budget and budget.expired():
                done, stop_reason = True, 'wall_time_budget'
                print(f'[budget] stopping optimizer updates at step {gs}; reserving time for validation/checkpoint', flush=True)
                break
            drop_rng, sub = jax.random.split(drop_rng)
            acc, l, _ = micro_jit(params, acc, batch, sub)
            loss_sum = loss_sum + l
            ia += 1
            if ia < t.accum_steps:
                continue
            lr_now = budget.learning_rate(gs, t.peak_lr, t.warmup_ratio, t.min_lr_ratio) if budget else float(sched(gs))
            if budget:
                params, ost, fin, gn = apply_jit(params, ost, acc, lr_now)
            else:
                params, ost, fin, gn = apply_jit(params, ost, acc)
            acc, ia = zero_fn(params), 0
            gs += 1
            lv = float(jax.device_get(loss_sum)) / t.accum_steps
            loss_sum = 0.0
            if not bool(jax.device_get(fin)):
                skipped += 1
                raise FloatingPointError(f"Non-finite gradient at step {gs}; refusing a skipped training run")
            if not math.isfinite(lv):
                raise FloatingPointError(f"Non-finite loss at step {gs}")
            if gs % t.log_every == 0:
                el = time.perf_counter() - t0
                eta = (total_steps - gs) * el / max(gs - start_step, 1)
                rec = dict(step=gs, epoch=ep, loss=lv, bpb_est=to_bpb(lv, bpt), grad_norm=float(jax.device_get(gn)),
                           finite=bool(jax.device_get(fin)), skipped=skipped, lr=lr_now, elapsed_s=el,
                           tokens_seen=gs * t.batch_size * t.accum_steps * t.seq_len,
                           wall_remaining_s=budget.remaining() if budget else None)
                log.log("train", **rec)
                print(f"[E{ep}] step={gs}/{total_steps} loss={lv:.4f} bpb~{rec['bpb_est']:.4f} gn={rec['grad_norm']:.3f} "
                      f"finite={rec['finite']} skipped={skipped} lr={rec['lr']:.2e} elapsed={el/60:.1f}m eta={eta/60:.1f}m",
                      flush=True)
            if t.eval_every and gs % t.eval_every == 0:
                vl, vb = evaluate(eval_jit, params, val_loader, t.eval_batches, bpt)
                best_bpb = min(best_bpb, vb)
                log.log("eval", step=gs, val_loss=vl, val_bpb_est=vb, partial=True)
                print(f"[EVAL] step={gs} val_loss={vl:.4f} val_bpb~{vb:.4f}", flush=True)
            if t.ckpt_every and gs % t.ckpt_every == 0 and ckpt.save(gs, params, ost):
                print(f"[ckpt] saved step {gs}", flush=True)
            if gs >= stop_at:
                done = True
                break
        # end of epoch (or early stop): full validation pass + checkpoint
        if gs == 0:
            raise RuntimeError('No optimizer step completed within the time budget')
        vl, vb = evaluate(eval_jit, params, val_loader, None, bpt)
        best_bpb = min(best_bpb, vb)
        log.log("eval", step=gs, epoch=ep, val_loss=vl, val_bpb_est=vb, partial=False, epoch_minutes=(time.perf_counter() - e0) / 60)
        print(f"\n===> epoch {ep}: {(time.perf_counter() - e0) / 60:.1f} min | val_loss={vl:.4f} val_bpb~{vb:.4f} <===\n", flush=True)
        ckpt.save(gs, params, ost)

    hours = (time.perf_counter() - t0) / 3600
    summary = dict(name=cfg.name, arch=cfg.arch, n_params=n_params, final_step=gs, skipped_steps=skipped,
                   best_val_bpb_est=best_bpb, hours=hours, run_dir=str(run_dir), stop_reason=stop_reason,
                   tokens_seen=gs * t.batch_size * t.accum_steps * t.seq_len,
                   completed_data_pass=gs >= total_steps)
    log.log("done", **summary)
    ckpt.mngr.close()
    print(f"[done] {hours:.2f} h | best val_bpb~{best_bpb:.4f} | skipped={skipped} | {run_dir}", flush=True)
    return summary
