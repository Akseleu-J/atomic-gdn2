from pathlib import Path

import pytest

from atomic_gdn2_model.config import ExperimentConfig

CONFIGS = sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_shipped_configs_load_and_roundtrip(path):
    cfg = ExperimentConfig.from_yaml(path)
    assert ExperimentConfig.from_dict(cfg.to_dict()) == cfg


def test_layer_pattern_is_3_gdn2_to_1_mla():
    cfg = ExperimentConfig.from_yaml(CONFIGS[0].parent / "hybrid_hier_1024_28l.yaml")
    types = cfg.model.layer_types
    assert len(types) == 28 and types.count("gdn2") == 21 and types.count("mla") == 7


def test_overrides_parse_types():
    cfg = ExperimentConfig.from_yaml(CONFIGS[0].parent / "smoke_cpu.yaml")
    new = cfg.with_overrides(["train.epochs=2", "train.peak_lr=1e-3", "arch=hybrid", "train.max_steps=null"])
    assert new.train.epochs == 2 and new.train.peak_lr == 1e-3 and new.arch == "hybrid" and new.train.max_steps is None


@pytest.mark.parametrize("bad", ["train.nope=1", "nosection.x=1", "arch=transformer", "train.seq_len=100", "oops"])
def test_bad_overrides_rejected(bad):
    cfg = ExperimentConfig.from_yaml(CONFIGS[0].parent / "smoke_cpu.yaml")
    with pytest.raises(ValueError):
        cfg.with_overrides([bad])


def test_unknown_yaml_keys_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"train": {"epochz": 1}})
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"bogus": 1})
