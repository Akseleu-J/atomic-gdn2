"""Model zoo: pure GDN-2 baseline and the GDN-2/MLA hybrid with hierarchical block-delta DAR."""
from __future__ import annotations

from ..config import ModelConfig
from .baseline import BaselineBlock, BaselineLM
from .common import SwiGLU
from .gdn2 import GDN2Sub
from .hybrid import LM, PaperBlock, PaperLayer, PaperDAR
from .mla import MLASub


def build_model(arch: str, cfg: ModelConfig, deterministic: bool):
    """Instantiate the LM for ``arch`` in {"baseline", "hybrid"}."""
    if arch == "baseline":
        return BaselineLM(cfg=cfg, n_blocks=cfg.num_blocks, gdn2_per_block=cfg.layers_per_block,
                          deterministic=deterministic)
    if arch == "hybrid":
        return LM(cfg=cfg, deterministic=deterministic)
    raise ValueError(f"unknown arch {arch!r}")


def count_params(params) -> int:
    import jax
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


__all__ = ["build_model", "count_params", "BaselineLM", "BaselineBlock", "LM", "PaperBlock", "PaperLayer", "PaperDAR", "GDN2Sub", "MLASub", "SwiGLU"]
