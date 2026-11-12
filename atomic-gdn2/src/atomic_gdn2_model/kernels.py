"""Kernel bindings.

All Pallas TPU kernels (the GDN-2 block-local-rescaling forward/backward) live in the external
**atomic-ops** package (``pip install atomic-ops``; import name ``atomic_gdn2``).  This repository
contains no kernel code - only the thin, cached binding below.
"""
from __future__ import annotations

from functools import lru_cache

try:
    from atomic_gdn2 import make_blr_trainable, make_cfg
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The GDN-2 kernels are provided by the `atomic-ops` package. Install it with:\n"
        "    pip install atomic-ops"
    ) from e


@lru_cache(maxsize=None)
def get_blr(bt: int, bs2: int, mb: int, scale: float, interpret: bool):
    """Return ``(blr_fn, kernel_cfg)``.

    ``blr_fn(q, k, v, w, b, g, h0=None) -> (o, h_final)`` takes head-major ``(B, H, L, D)``
    float32 inputs with ``L % bt == 0`` and carries a custom VJP.  ``kernel_cfg.g_max`` is the
    per-token clamp on ``|g|`` that keeps the BTL diagonal numerically valid.
    """
    cfg = make_cfg(interpret=interpret, bt=bt, bc=bt // 2, score_bs=bt, bs2=bs2, mb=mb)
    return make_blr_trainable(cfg, float(scale)), cfg
