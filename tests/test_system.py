import torch

from mtdb.core.system import MTDBSystem
from mtdb.utils import set_seed


def _make(cfg):
    set_seed(0)
    return MTDBSystem(cfg, input_dim=10, upper_dim=10, seed=0)


def test_step_and_interface(tiny_cfg):
    system = _make(tiny_cfg)
    for _ in range(10):
        out = system.step(torch.randn(10) * 0.2, learn=True, allow_memory_write=True)
    assert out.packet.z.shape == (system.spec.total_dim,)
    assert torch.isfinite(out.packet.z).all()
    assert out.packet.z[system.spec.slices()["aux"]].shape == (3,)
    assert system.wm.n_filled >= 0
    assert 0 <= out.spike_fraction <= 1


def test_memory_writes_and_eval_freeze(tiny_cfg):
    system = _make(tiny_cfg)
    for _ in range(30):
        out = system.step(torch.randn(10) * 0.2, learn=True, allow_memory_write=True)
    n_writes = system.wm.n_writes
    for _ in range(5):
        out = system.step(torch.randn(10) * 0.2, learn=False, allow_memory_write=False)
    assert system.wm.n_writes == n_writes
    assert out.gate is False


def test_checkpoint_roundtrip(tiny_cfg):
    system = _make(tiny_cfg)
    for _ in range(10):
        system.step(torch.randn(10) * 0.2, learn=True, allow_memory_write=True)
    state = system.state_dict()
    other = _make(tiny_cfg)
    other.load_state_dict(state)
    system._prev_s = None
    other._prev_s = None
    s = torch.randn(10) * 0.2
    a = system.step(s, learn=False, allow_memory_write=False)
    b = other.step(s, learn=False, allow_memory_write=False)
    assert torch.allclose(a.packet.z, b.packet.z, atol=1e-5)


def test_memory_disabled_still_has_interface(tiny_cfg):
    tiny_cfg.memory.enabled = False
    system = _make(tiny_cfg)
    out = system.step(torch.randn(10) * 0.2, learn=True, allow_memory_write=True)
    assert out.packet.z.shape == (system.spec.total_dim,)
    assert system.wm is None
    sl = system.spec.slices()
    assert out.packet.z[sl["wm"]].abs().sum() == 0
    assert out.packet.z[sl["attractor"]].abs().sum() == 0
