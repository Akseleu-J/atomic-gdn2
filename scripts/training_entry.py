"""Own the TPU for both preflight and training; invoked once in the isolated venv."""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--id', required=True)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--arch', choices=['baseline', 'hybrid'], required=True)
    p.add_argument('--seed', choices=[1, 2], required=True, type=int)
    p.add_argument('--deadline-unix', type=float, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    run_dir = root / 'runs' / args.id
    # mkdir is exclusive: even an interrupted run must never silently auto-resume.
    run_dir.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import yaml
    from atomic_gdn2_model.config import ExperimentConfig

    cfg_path = root / 'source' / 'atomic-gdn2' / 'configs' / (
        'baseline_gdn2_owt.yaml' if args.arch == 'baseline' else 'hybrid_hier_1024_28l.yaml')
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg['name'] = args.id
    cfg['train'].update(seed_init=args.seed, seed_data=42 + args.seed, seed_dropout=123 + args.seed)
    cfg['train'].update(batch_size=32, eval_every=1500, epochs=1)
    cfg['runtime'].update(require_tpu=True, shard=True, backend_wait_s=1,
                          wall_deadline_unix=args.deadline_unix, checkpoint_margin_s=600)
    cfg['data']['in_memory'] = False
    # Match optimizer step/token budgets and evaluation cadence between the two arms.
    candidates = [d for d in Path('/kaggle/input').rglob('train.bin') if (d.parent / 'val.bin').is_file()]
    candidates = [d for d in candidates if 'openwebtext' in str(d).lower()]
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one OpenWebText train.bin/val.bin pair; found {len(candidates)}')
    data_dir = candidates[0].parent
    cfg['data']['dir'] = str(data_dir)
    data = {}
    for split in ('train', 'val'):
        path = data_dir / f'{split}.bin'
        size = path.stat().st_size
        if size % 2 or size < 2 * cfg['train']['batch_size'] * (cfg['train']['seq_len'] + 1):
            raise ValueError(f'Invalid or insufficient uint16 data: {path} ({size} bytes)')
        tokens = np.memmap(path, dtype='<u2', mode='r')
        if split == 'train':
            cfg['data']['train_tokens'] = len(tokens)
        effective = min(len(tokens), cfg['data'][f'{split}_tokens'])
        if split == 'train' and effective != cfg['data']['train_tokens']:
            raise ValueError('Training split is shorter than the requested token budget')
        # Bound RAM while validating every token in the selected corpus, not just a random sample.
        for start in range(0, effective, 8_000_000):
            if int(tokens[start:min(start + 8_000_000, effective)].max()) >= cfg['model']['vocab_size']:
                raise ValueError(f'Out-of-vocabulary token in {path}')
        with path.open('rb') as f:
            first = f.read(1024 * 1024)
            f.seek(max(0, size - 1024 * 1024))
            last = f.read(1024 * 1024)
        data[split] = dict(path=str(path), bytes=size, available_tokens=len(tokens),
                           effective_tokens=effective, edge_sha256=hashlib.sha256(first + last).hexdigest())
        del tokens
    resolved = ExperimentConfig.from_dict(cfg)
    config_new = run_dir / 'config.yaml'
    config_new.write_text(resolved.to_yaml())
    # This is the ONLY process that calls JAX devices/default_backend. cli.main reuses it.
    import jax
    if jax.default_backend() != 'tpu' or jax.device_count() != 8 or jax.process_count() != 1:
        raise RuntimeError('Expected one JAX process with exactly 8 TPU devices; refusing CPU fallback')
    # Exercise forward/backward and data sharding on the allocated hardware before the expensive model.
    import jax.numpy as jnp
    from atomic_gdn2_model import distributed as dist
    from atomic_gdn2_model.models import build_model
    from atomic_gdn2_model.optim import path_to_str
    tiny_cfg = dataclasses.replace(resolved.model, vocab_size=256, d_model=256, n_heads=2,
        d_ff=512, num_blocks=2, layers_per_block=2, kv_lora_rank=64, d_latent=32, dropout_rate=0.0)
    mesh = dist.build_mesh()
    dist.set_mesh(mesh)
    tiny = build_model(args.arch, tiny_cfg, True)
    ids = jax.device_put(np.tile(np.arange(128, dtype=np.int32), (8, 1)), dist.batch_sharding(mesh))
    tiny_params = jax.jit(lambda key: tiny.init(key, ids)['params'], out_shardings=dist.replicated(mesh))(
        jax.random.PRNGKey(20261015))
    def next_token_ce(p):
        logits = tiny.apply({'params': p}, ids)
        labels = (ids + 1) % 256
        return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits, -1), labels[..., None], -1))

    value, grads = jax.jit(jax.value_and_grad(next_token_ce))(tiny_params)
    finite = bool(jnp.isfinite(value)) and all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree_util.tree_leaves(grads))
    if not finite:
        raise FloatingPointError('TPU kernel/model preflight has non-finite forward/backward values')
    dar_grads = [v for p, v in jax.tree_util.tree_flatten_with_path(grads)[0] if '/dar/' in path_to_str(p)]
    dar_max = max((float(jnp.max(jnp.abs(v))) for v in dar_grads), default=0.0)
    if args.arch == 'hybrid' and dar_max <= 0:
        raise RuntimeError('Connected DAR must receive a nonzero task gradient on the TPU preflight')
    runtime_probe = dict(backend='tpu', devices=8, data_sharding=True, forward_backward_finite=finite,
                         dar_gradient_max_abs=dar_max, loss=float(value), loss_type='next_token_cross_entropy',
                         cpu_parity_checked=False)
    (run_dir / 'runtime-preflight.json').write_text(json.dumps(runtime_probe, indent=2))
    print('[TPU PREFLIGHT PASS]', json.dumps(runtime_probe), flush=True)
    del value, grads, dar_grads, ids, tiny_params, tiny
    dist.set_mesh(None)
    jax.clear_caches()
    gc.collect()
    steps_per_epoch = ((data['train']['effective_tokens'] - 1) // (resolved.train.seq_len + 1)
                       // resolved.train.batch_size // resolved.train.accum_steps)
    provenance = dict(nb_id=args.id, arch=args.arch, seed=args.seed, pid=os.getpid(),
                      backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
                      started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      source=json.loads((root / 'source' / 'manifest.json').read_text()), data=data,
                      config_sha256=hashlib.sha256(config_new.read_bytes()).hexdigest(),
                      steps_per_epoch=steps_per_epoch, total_steps=steps_per_epoch * resolved.train.epochs,
                      time_budget_seconds=32040, finalization_reserve_seconds=600,
                      learning_rate_schedule='wall_time_with_first_100_step_warmup',
                      comparison_caveats=['Connected DAR is a new architecture; no previous quality result transfers',
                                          'Parameter counts and FLOPs differ between architectures',
                                          'Validation is limited to the available val.bin; bpb is an estimate'])
    (run_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print('[PREFLIGHT]', json.dumps(provenance), flush=True)
    from atomic_gdn2_model.cli import main as train
    train(['--config', str(config_new), '--run-dir', str(run_dir)])
    records = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text().splitlines()]
    done = [r for r in records if r['kind'] == 'done']
    if (len(done) != 1 or done[0]['final_step'] <= 0 or done[0]['skipped_steps']
            or done[0]['stop_reason'] not in ('wall_time_budget', 'data_complete')):
        raise RuntimeError('Training did not complete the requested time budget or data pass cleanly')
    (run_dir / 'SUCCESS.json').write_text(json.dumps(done[0], indent=2))


if __name__ == '__main__':
    main()
