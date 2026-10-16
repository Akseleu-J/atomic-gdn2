"""Typed, YAML-backed experiment configuration.

One :class:`ExperimentConfig` fully describes a run (architecture, data, optimisation,
runtime).  Configs live in ``configs/*.yaml`` and can be overridden from the CLI with
``--set section.field=value``.
"""
from __future__ import annotations

import dataclasses as dc
import math
import typing
from pathlib import Path
from typing import Any, Sequence

import jax.numpy as jnp
import yaml

ARCHS = ("baseline", "hybrid")


@dc.dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50257
    d_model: int = 1024
    n_heads: int = 8
    d_ff: int = 3584
    layers_per_block: int = 4
    num_blocks: int = 7
    dropout_rate: float = 0.1
    # GDN-2 (kernels come from the external `atomic-ops` package)
    gdn2_chunk_size: int = 128      # BLR chunk length `bt`; sequence length must be divisible by it
    gdn2_bs2: int = 32              # inner BLR block; limits |g| (g_max = 160 / bs2)
    gdn2_mb: int = 8                # ladder leaf size
    gdn2_g_init: float = 0.05       # softplus(bias) at init -> slow forgetting
    l2norm_qk: bool = True
    # MLA (hybrid only)
    rope_theta: float = 10000.0
    kv_lora_rank: int = 512
    q_lora_rank: int | None = None
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 64
    # hierarchical block-delta DAR (hybrid only)
    d_latent: int = 256
    layer_types_override: tuple[str, ...] = ()
    # misc
    tie_embeddings: bool = True
    use_bf16: bool = True           # bf16 compute, fp32 params
    kernel_interpret: bool = False  # Pallas interpret mode (CPU debugging only)

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def cdtype(self):
        return jnp.bfloat16 if self.use_bf16 else jnp.float32

    @property
    def pdtype(self):
        return jnp.float32

    @property
    def layer_types(self) -> tuple[str, ...]:
        if self.layer_types_override:
            return self.layer_types_override
        pattern = ("gdn2",) * (self.layers_per_block - 1) + ("mla",)
        return pattern * self.num_blocks

    @property
    def num_layers(self) -> int:
        return len(self.layer_types)


@dc.dataclass(frozen=True)
class DataConfig:
    dir: str = ""                    # empty -> first existing entry of `candidates`
    candidates: tuple[str, ...] = (
        "/kaggle/input/datasets/windmaple/openwebtext-gpt2",
        "/kaggle/input/openwebtext-gpt2",
        "/kaggle/input/windmaple/openwebtext-gpt2",
    )
    train_file: str = "train.bin"    # uint16 GPT-2 BPE token ids
    val_file: str = "val.bin"
    train_tokens: int = 1_500_000_000
    val_tokens: int = 50_000_000
    in_memory: bool = True           # copy the token slice to RAM (False -> np.memmap view)
    # Approximate UTF-8 bytes per GPT-2 token on OpenWebText; bpb is an ESTIMATE
    # (loss / ln2 / bytes_per_token), not an exact bits-per-byte measurement.
    bytes_per_token: float = 4.20


@dc.dataclass(frozen=True)
class TrainConfig:
    seq_len: int = 2048
    batch_size: int = 48             # global batch (micro-batch) size
    accum_steps: int = 1
    epochs: int = 3
    peak_lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.01
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    eval_every: int = 3000
    eval_batches: int = 100
    ckpt_every: int = 5000
    ckpt_keep: int = 2
    log_every: int = 50
    seed_init: int = 0
    seed_data: int = 42
    seed_dropout: int = 123
    max_steps: int | None = None     # early stop (smoke tests); LR schedule still spans all epochs


@dc.dataclass(frozen=True)
class RuntimeConfig:
    shard: bool = True               # data-parallel over all visible devices
    require_tpu: bool = True
    backend_wait_s: int = 30         # retry window while the TPU is still held by a previous process
    wall_deadline_unix: float | None = None
    checkpoint_margin_s: int = 600   # included in the outer job budget, reserved for final eval/checkpoint


@dc.dataclass(frozen=True)
class ExperimentConfig:
    name: str = "experiment"
    arch: str = "baseline"           # "baseline" (pure GDN-2) | "hybrid" (GDN-2/MLA + hierarchical DAR)
    model: ModelConfig = dc.field(default_factory=ModelConfig)
    data: DataConfig = dc.field(default_factory=DataConfig)
    train: TrainConfig = dc.field(default_factory=TrainConfig)
    runtime: RuntimeConfig = dc.field(default_factory=RuntimeConfig)

    def __post_init__(self):
        if self.arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}, got {self.arch!r}")
        m, t = self.model, self.train
        if m.d_model % m.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if t.seq_len % m.gdn2_chunk_size:
            raise ValueError(f"seq_len={t.seq_len} must be divisible by gdn2_chunk_size={m.gdn2_chunk_size}")
        if t.accum_steps < 1 or t.batch_size < 1 or t.epochs < 1:
            raise ValueError("accum_steps, batch_size, epochs must be >= 1")
        if not 0.0 <= m.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")
        if self.arch == "hybrid":
            if len(m.layer_types) % m.layers_per_block:
                raise ValueError("number of layers must be a multiple of layers_per_block")
            if m.d_model % 2 or m.qk_rope_head_dim % 2:
                raise ValueError("RoPE dims must be even")
        # g_max * bs2 / 2 < 88 is the BTL validity bound; make_cfg ties g_max = 160/bs2 -> 80.
        if math.isnan(t.peak_lr):
            raise ValueError("peak_lr is NaN")
        if self.runtime.checkpoint_margin_s < 0:
            raise ValueError('checkpoint_margin_s must be nonnegative')
        if self.runtime.wall_deadline_unix is not None and not math.isfinite(self.runtime.wall_deadline_unix):
            raise ValueError('wall_deadline_unix must be finite')

    # ---- (de)serialisation -------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return _plain(dc.asdict(self))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        d = dict(d)
        kwargs: dict[str, Any] = {}
        for f in dc.fields(cls):
            if f.name not in d:
                continue
            v = d.pop(f.name)
            sub = _SECTIONS.get(f.name)
            kwargs[f.name] = _build(sub, v, f.name) if sub else v
        if d:
            raise ValueError(f"unknown top-level config keys: {sorted(d)}")
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def with_overrides(self, overrides: Sequence[str]) -> "ExperimentConfig":
        """Apply ``["train.epochs=1", "arch=hybrid", ...]``; values are parsed as YAML scalars."""
        d = self.to_dict()
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"override must look like key=value, got {item!r}")
            key, raw = item.split("=", 1)
            node = d
            parts = key.strip().split(".")
            for p in parts[:-1]:
                if p not in node or not isinstance(node[p], dict):
                    raise ValueError(f"unknown config section {p!r} in {key!r}")
                node = node[p]
            if parts[-1] not in node:
                raise ValueError(f"unknown config key {key!r}")
            node[parts[-1]] = yaml.safe_load(raw)
        return ExperimentConfig.from_dict(d)


_SECTIONS = {"model": ModelConfig, "data": DataConfig, "train": TrainConfig, "runtime": RuntimeConfig}


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def _build(cls, values: dict[str, Any], section: str):
    if not isinstance(values, dict):
        raise ValueError(f"section {section!r} must be a mapping")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dc.fields(cls)}
    extra = set(values) - known
    if extra:
        raise ValueError(f"unknown keys in [{section}]: {sorted(extra)}")
    out = {}
    for k, v in values.items():
        h = hints[k]
        if isinstance(v, list) and "tuple" in str(h):
            v = tuple(v)
        elif h is float and isinstance(v, (int, str)) and not isinstance(v, bool):
            try:                      # PyYAML reads `1e-3` (no dot) as a string; accept it
                v = float(v)
            except ValueError as e:
                raise ValueError(f"[{section}] {k}: expected a number, got {v!r}") from e
        out[k] = v
    return cls(**out)
