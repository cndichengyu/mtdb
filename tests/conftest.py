import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtdb.config import ExperimentConfig


def _tiny() -> ExperimentConfig:
    cfg = ExperimentConfig.from_yaml(ROOT / "configs" / "debug.yaml")
    cfg.seed = 0
    cfg.device = "cpu"
    cfg.out_dir = "artifacts"
    cfg.run_name = "pytest"
    cfg.dynamics.n_units = 64
    cfg.dynamics.in_degree = 4
    cfg.dynamics.h_dim = 16
    cfg.dynamics.projection_fan_in = 8
    cfg.dynamics.noise_std = 0.0
    cfg.phase.n_clusters = 4
    cfg.phase.phase_noise = 0.0
    cfg.memory.wm_slots = 4
    cfg.memory.wm_read_dim = 8
    cfg.memory.attr_capacity = 16
    cfg.memory.attr_read_dim = 8
    cfg.memory.attr_topk = 4
    cfg.train.seq_len = 16
    cfg.train.n_train = 2
    cfg.train.n_eval = 1
    cfg.train.epochs = 1
    cfg.train.horizon = 2
    cfg.train.online_epochs = 1
    cfg.train.n_recall_train = 2
    cfg.train.n_recall_eval = 1
    cfg.train.ablations = False
    cfg.train.plot = False
    cfg.train.save_model = False
    cfg.consolidation.buffer_size = 4
    cfg.consolidation.window_len = 8
    cfg.consolidation.replay_passes = 1
    cfg.train.replay_stride = 8
    return cfg


@pytest.fixture
def tiny_cfg() -> ExperimentConfig:
    return _tiny()
