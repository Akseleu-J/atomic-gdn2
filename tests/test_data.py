import numpy as np
import pytest

from atomic_gdn2_model.config import DataConfig
from atomic_gdn2_model.data import Loader, find_data_dir, load_tokens


@pytest.fixture
def toks():
    return np.arange(5000, dtype=np.uint16)


def collect(it):
    return [np.asarray(b["input_ids"]) for b in it]


def test_shapes_and_label_shift(toks):
    L = Loader(toks, 16, 4, shuffle=False)
    b = next(iter(L.epoch(0)))
    assert b["input_ids"].shape == (4, 16) and b["labels"].shape == (4, 16)
    assert (np.asarray(b["labels"])[:, :-1] == np.asarray(b["input_ids"])[:, 1:]).all()


def test_epoch_order_depends_on_epoch_index_not_call_count(toks):
    L1, L2 = Loader(toks, 16, 4, seed=7), Loader(toks, 16, 4, seed=7)
    e0, e1 = collect(L1.epoch(0)), collect(L1.epoch(1))
    assert not all((a == b).all() for a, b in zip(e0, e1))            # epochs differ
    assert all((a == b).all() for a, b in zip(e1, collect(L2.epoch(1))))  # fresh loader, same epoch -> same order


def test_skip_equals_dropping_prefix(toks):
    L = Loader(toks, 16, 4, seed=3)
    full, skipped = collect(L.epoch(2)), collect(L.epoch(2, skip=5))
    assert len(skipped) == len(full) - 5 and all((a == b).all() for a, b in zip(full[5:], skipped))


def test_too_little_data():
    with pytest.raises(ValueError):
        Loader(np.arange(10, dtype=np.uint16), 16, 4)


def test_load_tokens_from_dir(tmp_path):
    np.arange(1000, dtype=np.uint16).tofile(tmp_path / "train.bin")
    np.arange(500, dtype=np.uint16).tofile(tmp_path / "val.bin")
    cfg = DataConfig(dir=str(tmp_path), train_tokens=800, val_tokens=300)
    tr, va = load_tokens(cfg)
    assert len(tr) == 800 and len(va) == 300 and find_data_dir(cfg) == tmp_path
    with pytest.raises(FileNotFoundError):
        find_data_dir(DataConfig(dir=str(tmp_path / "nope")))
