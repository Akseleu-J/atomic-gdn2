import importlib.util

import pytest


def pytest_collection_modifyitems(config, items):
    if importlib.util.find_spec("atomic_gdn2") is None:
        skip = pytest.mark.skip(reason="atomic-ops (atomic_gdn2) not installed")
        for it in items:
            if "kernel" in it.keywords:
                it.add_marker(skip)
