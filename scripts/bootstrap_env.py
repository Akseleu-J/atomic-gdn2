#!/usr/bin/env python3
"""Create an isolated venv with the pinned TPU stack and install this repo into it.

Replaces the ad-hoc venv/sys.path juggling of the original notebooks. Kaggle/Colab ship their own
(different) JAX; training therefore runs in a *subprocess* of the venv interpreter, so the notebook
kernel never imports JAX and never holds the TPU.

    python scripts/bootstrap_env.py                 # auto-detect platform, venv in <work_root>/venv_train
    python scripts/bootstrap_env.py --venv /tmp/v   # custom location
Prints the venv python path on the last line.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def detect_platform() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_TPU_ADDR"):
        return "colab"
    has_kaggle, has_colab = os.path.isdir("/kaggle/working"), os.path.isdir("/content")
    if has_kaggle and not has_colab:
        return "kaggle"
    if has_colab and not has_kaggle:
        return "colab"
    cwd = os.getcwd()
    if cwd.startswith("/kaggle"):
        return "kaggle"
    if cwd.startswith("/content"):
        return "colab"
    return "local"


def work_root(platform: str) -> Path:
    return {"kaggle": Path("/kaggle/working"), "colab": Path("/content")}.get(platform, Path.cwd())


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), **kw)


def make_venv(venv: Path) -> Path:
    py, pip = venv / "bin" / "python", venv / "bin" / "pip"
    if py.exists() and pip.exists() and run([py, "-c", "import sys"], capture_output=True).returncode == 0:
        return py
    shutil.rmtree(venv, ignore_errors=True)
    r = run([sys.executable, "-m", "virtualenv", "--python", sys.executable, venv], capture_output=True, text=True)
    if r.returncode != 0 or not pip.exists():      # virtualenv missing -> stdlib venv + get-pip
        shutil.rmtree(venv, ignore_errors=True)
        run([sys.executable, "-m", "venv", "--without-pip", venv], check=True)
        get_pip = Path("/tmp/get-pip.py")
        urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip)
        run([py, get_pip], check=True)
    return py


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--venv", default=None)
    ap.add_argument("--requirements", default=str(REPO / "requirements-tpu.txt"))
    ap.add_argument("--no-install-repo", action="store_true")
    a = ap.parse_args()

    platform = detect_platform()
    venv = Path(a.venv) if a.venv else work_root(platform) / "venv_train"
    print(f"[bootstrap] platform={platform} venv={venv}", flush=True)
    py = make_venv(venv)
    run([py, "-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools", "wheel"], check=True)
    run([py, "-m", "pip", "install", "--no-cache-dir", "-r", a.requirements], check=True)
    if not a.no_install_repo:
        run([py, "-m", "pip", "install", "--no-cache-dir", "-e", REPO], check=True)
    run([py, "-c",
         "import jax, flax, optax, atomic_gdn2, atomic_gdn2_model; "
         "print(f'jax={jax.__version__} flax={flax.__version__} optax={optax.__version__} "
         "atomic-ops={atomic_gdn2.__version__} atomic-gdn2-model={atomic_gdn2_model.__version__}')"], check=True)
    print(py)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
