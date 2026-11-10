"""Small shared utilities: seeding, device handling, logging, running stats."""

from __future__ import annotations

import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    name = (name or "cpu").lower()
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    if name in ("auto", ""):
        # The sparse gather/phase/gating kernels are optimised for CPU and CUDA;
        # MPS is opt-in because float64 RLS state and repeated int64 gathers are
        # not reliably faster there.
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str = "mtdb", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def make_run_dir(out_dir: str | Path, run_name: str) -> Path:
    run_dir = Path(out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def count_parameters(tensors) -> int:
    total = 0
    for t in tensors:
        if torch.is_tensor(t):
            total += t.numel()
    return total


# ---------------------------------------------------------------------------
# Running normalizer used by the surprise gate
# ---------------------------------------------------------------------------
class RunningNorm:
    """Exponential moving mean/variance for surprise normalisation."""

    def __init__(self, decay: float = 0.99, eps: float = 1e-6):
        self.decay = decay
        self.eps = eps
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def update(self, value: float) -> None:
        value = float(value)
        self.count += 1
        if self.count == 1:
            self.mean = value
            self.var = max(value * value, self.eps)
            return
        self.mean = self.decay * self.mean + (1.0 - self.decay) * value
        self.var = self.decay * self.var + (1.0 - self.decay) * (value - self.mean) ** 2

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, self.eps))

    def normalize(self, value: float) -> float:
        return float((value - self.mean) / (self.std + self.eps))

    def threshold(self, k: float, floor: float = 0.0) -> float:
        """Raw-error threshold: mean + k * std, with optional absolute floor on z."""
        return self.mean + max(k, floor) * self.std


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0
