#!/usr/bin/env python3
"""Train MTDB on your own feature stream — minimal runnable example.

No synthetic-world classes are required: you provide arrays of shape [N, T, F],
the adapter derives prediction targets, z-scores using training statistics, and
the normal 3-stage MTDB pipeline runs unchanged.

Examples
--------
    # toy stream generated in-memory, tiny config (~5-10 s on CPU)
    python examples/custom_stream.py

    # your own .npz with at least 'train' and 'eval' arrays
    python examples/custom_stream.py --data data/my_stream.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from mtdb.config import ExperimentConfig
from mtdb.tasks.array import make_streams_from_arrays, streams_from_npz
from mtdb.training.trainer import MTDBTrainer
from mtdb.utils import get_logger, set_seed


def make_toy_arrays(seed: int = 0):
    """A tiny two-regime stream: damped oscillator + AR noise + sparse events."""
    rng = np.random.default_rng(seed)

    def one_episode(T: int, regime: str) -> np.ndarray:
        x = np.zeros((T, 8), dtype=np.float32)
        pos, vel = 0.0, 0.05
        phase = rng.uniform(0, 2 * np.pi)
        for t in range(T):
            accel = rng.normal(0, 0.002)
            if regime == "B":
                vel = -0.9 * vel + accel + 0.01 * np.sin(0.1 * t)
            else:
                vel = 0.98 * vel + accel
            pos = np.clip(pos + vel, -1.0, 1.0)
            phase += 0.08 + 0.01 * rng.normal()
            x[t, 0] = pos
            x[t, 1] = vel
            x[t, 2] = np.sin(phase)
            x[t, 3] = np.cos(phase)
            x[t, 4] = float(rng.random() < 0.03)
            x[t, 5] = 0.5 + 0.4 * rng.normal() if x[t, 4] else 0.0
            x[t, 6] = rng.normal(0, 0.05)
            x[t, 7] = np.clip(pos * pos, 0, 1)
        return x

    train = np.stack([one_episode(64, "A") for _ in range(12)])
    eval_ = np.stack([one_episode(64, "A") for _ in range(3)])
    train_b = np.stack([one_episode(64, "B") for _ in range(6)])
    eval_b = np.stack([one_episode(64, "B") for _ in range(2)])
    return train, eval_, train_b, eval_b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="optional .npz with train/eval arrays")
    ap.add_argument("--config", default=str(ROOT / "configs" / "debug.yaml"))
    ap.add_argument("--run-name", default="example_custom")
    ap.add_argument("--no-normalize", action="store_true")
    args = ap.parse_args()

    if args.data:
        streams, meta = streams_from_npz(
            args.data, horizon=5, normalize=not args.no_normalize
        )
    else:
        train, eval_, train_b, eval_b = make_toy_arrays()
        streams, meta = make_streams_from_arrays(
            train, eval_, train_b, eval_b, horizon=5,
            normalize=not args.no_normalize,
        )

    cfg = ExperimentConfig.from_yaml(args.config)
    # small overrides so this example is quick; remove them for a real run
    cfg.run_name = args.run_name
    cfg.dynamics.n_units = 256
    cfg.dynamics.in_degree = 8
    cfg.train.epochs = 1
    cfg.train.online_epochs = 1
    cfg.train.stress_online_epochs = 1
    cfg.train.run_recall = False
    cfg.train.ablations = False
    cfg.train.forgetting_stress = meta["has_regime_b"]
    cfg.train.plot = False
    cfg.train.save_model = False

    set_seed(cfg.seed)
    logger = get_logger("mtdb.example")
    logger.info(
        "custom stream: %d train / %d eval sequences, F=%d, regime-B=%s",
        meta["n_train"], meta["n_eval"], meta["input_dim"], meta["has_regime_b"],
    )
    trainer = MTDBTrainer(cfg, streams=streams)
    results = trainer.run()
    stage1 = results["stage1"]["prediction"]
    print("\n=== custom-stream example summary ===")
    print(f"input dim      : {results['input_dim']}")
    print(f"interface dim  : {results['interface_dim']}")
    print(f"stage-1 MSE    : {stage1['mse']:.5f} (NMSE {stage1['nmse']:.3f})")
    if "stage2" in results:
        s2 = results["stage2"]
        print(f"online B gain  : {s2['adaptation_gain_B']:+.5f}")
        print(f"sleep A gain   : {s2['sleep_gain_A']:+.5f}")
    print(f"artifacts      : {trainer.run_dir}")


if __name__ == "__main__":
    main()
