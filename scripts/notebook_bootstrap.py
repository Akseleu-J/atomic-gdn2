"""Notebook host: standard library only. Never imports or initializes JAX."""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path


def launch(nb_id, arch, seed, payload_b64, payload_sha256, wall_seconds=32040):
    deadline_unix = time.time() + wall_seconds
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('This notebook requires Kaggle')
    # Dropping sys.modules does not release native TPU resources: require a fresh session.
    if any(m == 'jax' or m.startswith('jax.') or m == 'libtpu' for m in sys.modules):
        raise RuntimeError('JAX already imported in Jupyter. Restart this session before Run All.')
    root = Path('/kaggle/working') / nb_id
    root.mkdir(exist_ok=False)
    stage = 'extract'
    try:
        raw = base64.b64decode(payload_b64)
        if hashlib.sha256(raw).hexdigest() != payload_sha256:
            raise ValueError('Source payload checksum mismatch')
        source = root / 'source'
        source.mkdir()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for member in z.infolist():
                target = (source / member.filename).resolve()
                if not target.is_relative_to(source.resolve()):
                    raise ValueError('Unsafe archive entry')
            z.extractall(source)
        manifest = json.loads((source / 'manifest.json').read_text())
        for name, digest in manifest['files'].items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Source file checksum mismatch: {name}')
        stage = 'network'
        import urllib.request
        with urllib.request.urlopen('https://pypi.org/simple/pip/', timeout=30) as response:
            if response.status != 200:
                raise RuntimeError('PyPI connectivity check failed; enable Internet in Settings')
        stage = 'venv'
        venv = root / 'venv'
        py = venv / 'bin' / 'python'

        def checked(cmd):
            print('[SETUP]', ' '.join(map(str, cmd)), flush=True)
            remaining = deadline_unix - time.time()
            if remaining <= 0:
                raise TimeoutError('The total job time budget expired during setup')
            subprocess.run(list(map(str, cmd)), check=True, timeout=min(2400, remaining))

        checked([sys.executable, '-m', 'venv', '--without-pip', venv])
        checked([sys.executable, '-m', 'pip', '--python', py, 'install', 'pip', 'setuptools', 'wheel'])
        stage = 'dependencies'
        checked([py, '-m', 'pip', 'install', '--only-binary=:all:', '-r', source / 'requirements.lock'])
        for repo in ('atomic-ops', 'atomic-gdn2'):
            checked([py, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', source / repo])
        checked([py, '-m', 'pip', 'check'])
        with (root / 'pip-freeze.txt').open('w') as f:
            subprocess.run([str(py), '-m', 'pip', 'freeze'], stdout=f, check=True,
                           timeout=max(1., deadline_unix - time.time()))
        stage = 'training'
        env = os.environ.copy()
        env['JAX_PLATFORMS'] = 'tpu'
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONNOUSERSITE'] = '1'
        env['USE_SHARDING'] = '1'
        env.pop('PYTHONPATH', None)
        cmd = [str(py), '-u', str(source / 'training_entry.py'), '--id', nb_id,
               '--root', str(root), '--arch', arch, '--seed', str(seed),
               '--deadline-unix', str(deadline_unix)]
        print('[RUN]', ' '.join(cmd), flush=True)
        with (root / 'training.log').open('w', buffering=1) as log:
            with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1) as process:
                timed_out = threading.Event()

                def enforce_deadline():
                    if process.poll() is None:
                        timed_out.set()
                        try:
                            process.kill()
                        except OSError:
                            pass

                watchdog = threading.Timer(max(0., deadline_unix - time.time()), enforce_deadline)
                watchdog.daemon = True
                watchdog.start()
                try:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                    rc = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=30)
                    raise
                finally:
                    watchdog.cancel()
                if timed_out.is_set():
                    raise TimeoutError('Hard job limit reached: 8h54m; see preserved logs/checkpoints')
        if rc:
            raise RuntimeError(f'Training child failed: exit code {rc}; see training.log')
        if not (root / 'runs' / nb_id / 'SUCCESS.json').is_file():
            raise RuntimeError('Child exited without a verified full-training success artifact')
    except BaseException:
        (root / 'FAILURE.json').write_text(json.dumps(dict(stage=stage, traceback=traceback.format_exc()), indent=2))
        raise
    finally:
        # Keep configs, logs, provenance, source and checkpoints; remove install artifacts from Kaggle outputs.
        venv = root / 'venv'
        if venv.is_dir():
            shutil.rmtree(venv)
