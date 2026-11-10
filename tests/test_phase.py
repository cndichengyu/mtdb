import math

import torch

from mtdb.core.dynamics import MultiTimescaleDynamics
from mtdb.core.phase import PhaseSyncRouter
from mtdb.utils import set_seed


def test_phase_bounds_and_features(tiny_cfg):
    set_seed(0)
    dyn = MultiTimescaleDynamics(tiny_cfg.dynamics, input_dim=10)
    phase = PhaseSyncRouter(
        tiny_cfg.phase,
        dyn.n_units,
        dyn.pre_idx,
        dyn.group_ids,
        input_dim=10,
        dt_ms=tiny_cfg.dynamics.dt_ms,
    )
    for _ in range(20):
        phase.step(torch.randn(10) * 0.2)
        feat, sync, bins = phase.features(dyn.x)
    assert phase.theta.min() >= 0 and phase.theta.max() < 2 * math.pi + 1e-5
    assert feat.shape == (3 * tiny_cfg.phase.n_clusters,)
    assert 0.0 <= sync <= 1.0
    assert bins.shape == (dyn.n_units,)
