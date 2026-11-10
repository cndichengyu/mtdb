from mtdb.config import ExperimentConfig, apply_overrides


def test_yaml_presets_load():
    for name in ("debug", "standard", "mvp"):
        cfg = ExperimentConfig.from_yaml(f"configs/{name}.yaml")
        assert cfg.dynamics.n_units > 0
        assert len(cfg.dynamics.tau_groups_ms) == 3
        assert cfg.n_interface_dims(10) == (
            cfg.dynamics.h_dim + 3 * cfg.phase.n_clusters
            + cfg.memory.wm_read_dim + cfg.memory.attr_read_dim + 20 + 3
        )


def test_overrides_and_roundtrip(tiny_cfg, tmp_path):
    apply_overrides(tiny_cfg, ["dynamics.n_units=32", "train.epochs=2"])
    assert tiny_cfg.dynamics.n_units == 32
    assert tiny_cfg.train.epochs == 2
    path = tmp_path / "cfg.yaml"
    tiny_cfg.save(path)
    loaded = ExperimentConfig.from_yaml(path)
    assert loaded.to_dict() == tiny_cfg.to_dict()


def test_ablation_flags_off(tiny_cfg):
    tiny_cfg.train.ablations = False
    assert tiny_cfg.train.ablations is False
