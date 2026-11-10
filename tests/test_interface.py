import pytest
import torch

from mtdb.core.interface import InterfaceSpec, pack_interface


def test_interface_slices():
    spec = InterfaceSpec(
        h_dim=8, phase_dim=6, wm_dim=4, attr_dim=4, n_clusters=2,
        input_dim=5, include_sensory=True,
    )
    sl = spec.slices()
    assert sl["h"] == slice(0, 8)
    assert sl["phase"] == slice(8, 14)
    assert sl["wm"] == slice(14, 18)
    assert sl["attractor"] == slice(18, 22)
    assert sl["input"] == slice(22, 27)
    assert sl["delta"] == slice(27, 32)
    assert sl["aux"] == slice(32, 35)
    s = torch.arange(5, dtype=torch.float32)
    z = pack_interface(
        spec,
        h=torch.zeros(8),
        phase_features=torch.zeros(6),
        wm_read=torch.zeros(4),
        attractor_read=torch.zeros(4),
        surprise=0.5,
        sync=0.2,
        confidence=0.7,
        s=s,
        prev_s=s - 1,
    )
    assert z.shape == (spec.total_dim,)
    assert z[spec.slices()["input"]].tolist() == pytest.approx(s.tolist())
    assert z[spec.slices()["delta"]].tolist() == pytest.approx([1.0] * 5)
    assert z[spec.slices()["aux"]].tolist() == pytest.approx([0.5, 0.2, 0.7])


def test_interface_without_sensory():
    spec = InterfaceSpec(h_dim=8, phase_dim=6, wm_dim=4, attr_dim=4, n_clusters=2,
                         include_sensory=False)
    assert spec.total_dim == 8 + 6 + 4 + 4 + 3
    z = pack_interface(spec, torch.zeros(8), torch.zeros(6), torch.zeros(4),
                       torch.zeros(4), 0.0, 0.0, 0.0)
    assert z.shape == (spec.total_dim,)


def test_interface_mismatch_raises():
    spec = InterfaceSpec(h_dim=8, phase_dim=6, wm_dim=4, attr_dim=4, n_clusters=2,
                         include_sensory=False)
    with pytest.raises(ValueError):
        pack_interface(spec, torch.zeros(7), torch.zeros(6), torch.zeros(4),
                       torch.zeros(4), 0.0, 0.0, 0.0)
