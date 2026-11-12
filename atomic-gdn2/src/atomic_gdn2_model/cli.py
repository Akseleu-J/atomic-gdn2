"""``python -m atomic_gdn2_model.cli --config configs/hybrid_hier_1024_28l.yaml``"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="atomic-gdn2-train", description="Train a GDN-2 language model (baseline or hybrid).")
    p.add_argument("--config", required=True, help="YAML experiment config (see configs/)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set train.epochs=1 (repeatable)")
    p.add_argument("--run-dir", default=None, help="output dir (default: $ATOMIC_LM_WORK_DIR or ./runs / <name>)")
    p.add_argument("--resume-from", default=None, help="directory with orbax checkpoints to continue from")
    p.add_argument("--print-config", action="store_true", help="print the resolved config and exit")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    cfg = ExperimentConfig.from_yaml(a.config).with_overrides(a.overrides)
    if a.print_config:
        print(cfg.to_yaml())
        return 0
    run_dir = Path(a.run_dir) if a.run_dir else Path(os.environ.get("ATOMIC_LM_WORK_DIR", "runs")) / cfg.name
    from .trainer import run  # defer the heavy jax import until after arg parsing
    run(cfg, run_dir, resume_from=a.resume_from)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
