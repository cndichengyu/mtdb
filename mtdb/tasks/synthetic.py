"""Synthetic multimodal streams used to validate the base.

Two worlds are provided:

1. ``MultimodalWorld`` — a partially observable dynamical scene with a moving
   object (hidden velocity), a slow audio oscillator, touch events and object
   identity changes. Observation dim = 10:
       [pos_x, pos_y, audio_sin, audio_cos, contact, force, obj_0..obj_3]
   The agent only sees positions, so next-state prediction must integrate
   hidden velocity over time — this is where multiple time constants help.
   Anomalies are transient sensor corruptions (teleport / audio glitch / wrong
   object) used to test the surprise signal.

2. ``DelayedRecallWorld`` — a cue is shown for a few ticks, then a long delay
   with distractors; the target is the cue class throughout, and accuracy is
   measured only at the end of the delay. This tests long-horizon memory.

Both worlds emit dicts of torch tensors: x [T, F], y [T, F], plus metadata.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

FEATURE_DIM = 10
N_OBJECTS = 4
N_CLASSES = 8


class MultimodalWorld:
    def __init__(
        self,
        horizon: int = 5,
        anomaly_rate: float = 0.0,
        anomaly_window: int = 2,
        seed: int = 0,
        dt: float = 0.01,
    ):
        self.horizon = int(horizon)
        self.anomaly_rate = float(anomaly_rate)
        self.anomaly_window = int(anomaly_window)
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)

    def _episode(
        self, seq_len: int, regime: str, anomalies: bool
    ) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(int(self.rng.integers(0, 2 ** 31 - 1)))
        T = int(seq_len)
        pos = rng.uniform(-0.7, 0.7, size=2)
        vel = rng.normal(0.0, 0.03, size=2)
        audio_phase = rng.uniform(0, 2 * np.pi)
        audio_base = rng.uniform(0.5, 1.2)
        audio_wobble = rng.uniform(0.0, 0.5)
        obj = int(rng.integers(0, N_OBJECTS))
        seg_len = int(rng.integers(18, 36))
        seg_t = 0
        contact = 0.0
        force = 0.0

        xs = np.zeros((T, FEATURE_DIM), dtype=np.float32)
        anomaly_mask = np.zeros(T, dtype=bool)
        corruption_kind = {}
        if anomalies and self.anomaly_rate > 0:
            t = int(rng.integers(6, max(7, T // 3)))
            while t < T - 2:
                # 0 teleport, 1 audio glitch, 2 wrong object,
                # 3 position freeze, 4 audio freeze
                kind = int(rng.integers(0, 5))
                for k in range(self.anomaly_window):
                    if t + k < T:
                        corruption_kind[t + k] = kind
                t += int(rng.integers(18, 40))

        for t in range(T):
            # --- latent dynamics -----------------------------------------
            if regime == "B":
                vel = -0.85 * vel + rng.normal(0.0, 0.012, size=2)
            else:
                vel = 0.98 * vel + rng.normal(0.0, 0.012, size=2)
            pos = pos + vel
            for d in range(2):
                if pos[d] > 1.0:
                    pos[d] = 1.0
                    vel[d] = -0.9 * vel[d]
                elif pos[d] < -1.0:
                    pos[d] = -1.0
                    vel[d] = -0.9 * vel[d]

            freq = audio_base + audio_wobble * np.sin(0.02 * t + audio_base)
            audio_phase = audio_phase + 2 * np.pi * freq * self.dt
            audio_sin, audio_cos = np.sin(audio_phase), np.cos(audio_phase)

            seg_t += 1
            if seg_t >= seg_len:
                obj = (obj + 1 + int(rng.integers(0, N_OBJECTS - 1))) % N_OBJECTS
                seg_len = int(rng.integers(18, 36))
                seg_t = 0

            if contact > 0.5:
                contact = 1.0 if rng.random() < 0.8 else 0.0
                force = float(np.clip(force + rng.normal(0.0, 0.05), 0.0, 1.0))
            else:
                contact = 1.0 if rng.random() < 0.05 else 0.0
                force = 0.15 if contact else 0.0

            obs = np.zeros(FEATURE_DIM, dtype=np.float32)
            obs[0:2] = pos
            obs[2] = audio_sin
            obs[3] = audio_cos
            obs[4] = contact
            obs[5] = force
            obs[6 + obj] = 1.0

            if t in corruption_kind:
                anomaly_mask[t] = True
                kind = corruption_kind[t]
                prev = xs[t - 1] if t > 0 else obs
                if kind == 0:      # teleport
                    obs[0:2] = rng.uniform(-1.0, 1.0, size=2)
                elif kind == 1:    # audio glitch
                    obs[2:4] = rng.normal(0.0, 1.0, size=2)
                elif kind == 2:    # wrong object
                    obs[6:10] = 0.0
                    obs[6 + int(rng.integers(0, N_OBJECTS))] = 1.0
                elif kind == 3:    # position freeze (dynamics violation, small input delta)
                    obs[0:2] = prev[0:2]
                else:              # audio freeze
                    obs[2:4] = prev[2:4]
            xs[t] = obs

        # --- targets: observation `horizon` ticks ahead -------------------
        idx = np.arange(T) + self.horizon
        idx = np.clip(idx, 0, T - 1)
        ys = xs[idx].copy()
        return {
            "x": torch.from_numpy(xs),
            "y": torch.from_numpy(ys),
            "anomaly": torch.from_numpy(anomaly_mask),
            "regime": regime,
        }

    def sample(
        self, seq_len: int = 96, regime: str = "A", anomalies: bool = False
    ) -> Dict[str, torch.Tensor]:
        return self._episode(seq_len, regime, anomalies)


class DelayedRecallWorld:
    def __init__(
        self,
        n_classes: int = N_CLASSES,
        cue_len: int = 3,
        delay: int = 40,
        tail: int = 10,
        seed: int = 0,
    ):
        self.n_classes = int(n_classes)
        self.cue_len = int(cue_len)
        self.delay = int(delay)
        self.tail = int(tail)
        self.feature_dim = FEATURE_DIM
        self.rng = np.random.default_rng(seed)

    @property
    def seq_len(self) -> int:
        return self.cue_len + self.delay + self.tail

    def sample(self, seq_len: Optional[int] = None) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(int(self.rng.integers(0, 2 ** 31 - 1)))
        T = int(seq_len or self.seq_len)
        cue = int(rng.integers(0, self.n_classes))
        x = np.zeros((T, FEATURE_DIM), dtype=np.float32)
        y = np.zeros((T, FEATURE_DIM), dtype=np.float32)
        y[:, cue] = 1.0
        x[: self.cue_len, cue] = 1.0
        # distractor noise in the last two dimensions
        x[:, self.n_classes :] = rng.normal(0.0, 0.2, size=(T, FEATURE_DIM - self.n_classes))
        eval_mask = np.zeros(T, dtype=bool)
        eval_mask[max(0, T - self.tail):] = True
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "eval_mask": torch.from_numpy(eval_mask),
            "cue": torch.tensor(cue),
        }


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def make_prediction_set(
    world: MultimodalWorld,
    n: int,
    seq_len: int,
    regime: str = "A",
    anomalies: bool = False,
) -> List[Dict[str, torch.Tensor]]:
    return [
        world.sample(seq_len=seq_len, regime=regime, anomalies=anomalies)
        for _ in range(n)
    ]


def make_recall_set(
    world: DelayedRecallWorld, n: int
) -> List[Dict[str, torch.Tensor]]:
    return [world.sample() for _ in range(n)]


def stack_windows(seqs: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """Stack a list of variable-free sequences' x into [N, T, F]."""
    return torch.stack([s["x"] for s in seqs], dim=0)
