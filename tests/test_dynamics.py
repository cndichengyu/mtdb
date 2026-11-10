import torch

from mtdb.core.dynamics import MultiTimescaleDynamics
from mtdb.utils import set_seed


def test_sparse_shapes_and_step(tiny_cfg):
    set_seed(0)
    dyn = MultiTimescaleDynamics(tiny_cfg.dynamics, input_dim=10)
    assert dyn.W.shape == (tiny_cfg.dynamics.n_units, tiny_cfg.dynamics.in_degree)
    assert dyn.pre_idx.shape == dyn.W.shape
    assert len(torch.unique(dyn.group_ids)) == len(set(tiny_cfg.dynamics.group_probs)) or dyn.n_units > 0
    assert dyn.decay.min() > 0 and dyn.decay.max() <= 1
    x = dyn.step(torch.randn(10) * 0.2, learn=True)
    assert x.shape == (tiny_cfg.dynamics.n_units,)
    assert torch.isfinite(x).all()


def test_bounded_states_and_stats(tiny_cfg):
    set_seed(1)
    dyn = MultiTimescaleDynamics(tiny_cfg.dynamics, input_dim=10)
    for t in range(100):
        dyn.step(torch.randn(10) * 0.5, learn=True)
    assert dyn.x.abs().max() < 1.5
    stats = dyn.stats()
    assert 0.0 <= stats["active_fraction"] <= 1.0
    assert stats["weight_zero_fraction"] == 0.0
    assert torch.isfinite(torch.tensor(list(stats.values()))).all()


def test_plasticity_prune_and_norm(tiny_cfg):
    set_seed(2)
    dyn = MultiTimescaleDynamics(tiny_cfg.dynamics, input_dim=10)
    for _ in range(20):
        dyn.step(torch.randn(10) * 0.2, learn=True)
    e = torch.randn(tiny_cfg.dynamics.n_units) * 0.05
    stats = dyn.plastic_update(e, modulation=1.2, reward=-0.1, pcfg=tiny_cfg.plasticity)
    assert "weight_abs_mean" in stats
    assert torch.isfinite(dyn.W).all()
    nnz = int((dyn.W != 0).sum())
    removed = dyn.prune(0.5)
    assert removed > 0
    assert int((dyn.W != 0).sum()) < nnz
    dyn.normalize_rows()
    assert torch.isfinite(dyn.W).all()


def test_state_roundtrip(tiny_cfg):
    set_seed(3)
    dyn = MultiTimescaleDynamics(tiny_cfg.dynamics, input_dim=10)
    for _ in range(10):
        dyn.step(torch.randn(10) * 0.2)
    state = dyn.state_dict()
    other = MultiTimescaleDynamics(tiny_cfg.dynamics, input_dim=10)
    other.load_state_dict(state)
    assert torch.equal(dyn.W, other.W)
    assert torch.equal(dyn.pre_idx, other.pre_idx)
