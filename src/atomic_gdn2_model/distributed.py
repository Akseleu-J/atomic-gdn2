"""Device mesh, shardings and backend bring-up (single-axis data parallelism)."""
from __future__ import annotations

import contextlib
import time

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

AXIS_NAME = "data"
_MESH: Mesh | None = None


def get_mesh() -> Mesh | None:
    return _MESH


def set_mesh(mesh: Mesh | None) -> None:
    """Models read the mesh at trace time (for ``shard_map`` around Pallas kernels)."""
    global _MESH
    _MESH = mesh


@contextlib.contextmanager
def use_mesh(mesh: Mesh | None):
    prev = _MESH
    set_mesh(mesh)
    try:
        yield mesh
    finally:
        set_mesh(prev)


def build_mesh() -> Mesh:
    return Mesh(np.array(jax.devices()), axis_names=(AXIS_NAME,))


def replicated(mesh: Mesh | None):
    return None if mesh is None else NamedSharding(mesh, P())


def batch_sharding(mesh: Mesh | None):
    return None if mesh is None else NamedSharding(mesh, P(AXIS_NAME, None))


def wait_for_backend(expect_tpu: bool = True, max_wait_s: int = 30):
    """Initialise JAX, retrying while a previous process still holds the TPU (vfio busy)."""
    last = None
    for i in range(max(1, max_wait_s)):
        try:
            backend, n = jax.default_backend(), jax.device_count()
            break
        except RuntimeError as e:
            last = e
            if any(s in str(e) for s in ("vfio", "busy", "TPU initialization")):
                if i == 0:
                    print(f"[backend] TPU busy, retrying for up to {max_wait_s}s ...", flush=True)
                time.sleep(1.0)
                continue
            raise
    else:
        raise RuntimeError("TPU stayed busy - disconnect/restart the runtime") from last
    if expect_tpu and backend != "tpu":
        raise RuntimeError(f"TPU required (runtime.require_tpu=true) but backend={backend!r}")
    return backend, n


def shard_map_auto(inner_fn, n_inputs: int):
    """Wrap ``inner_fn`` (all args/outputs ``(B, H, L, D)``) in ``shard_map`` over batch, else heads.

    Falls back to a plain call when no mesh is active or neither axis divides the device count.
    """
    mesh = get_mesh()
    if mesh is None:
        return inner_fn
    n_dev = mesh.shape[AXIS_NAME]

    def wrapped(*args):
        B, H = args[0].shape[0], args[0].shape[1]
        if B % n_dev == 0:
            spec = P(AXIS_NAME, None, None, None)
        elif H % n_dev == 0:
            spec = P(None, AXIS_NAME, None, None)
        else:
            return inner_fn(*args)
        return jax.shard_map(inner_fn, mesh=mesh, in_specs=(spec,) * n_inputs,
                             out_specs=spec, check_vma=False)(*args)

    return wrapped
