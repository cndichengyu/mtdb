"""Memory substrate (MTDB section C): working memory, attractor memory, surprise gate.

Design notes
------------
* Working memory is a small set of persistent slots (default 32) written only
  on surprising events; it is the "current situation" buffer.
* Long-term memory is an attractor memory: stored prototypes are stable states
  retrieved by pattern completion. Writes are novelty-gated and rate-limited.
* Surprise is the normalized one-step self-prediction error of the base. It
  gates writes, drives attention in the upper layer, and modulates plasticity.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from ..config import MemoryConfig


# ---------------------------------------------------------------------------
# Surprise gate
# ---------------------------------------------------------------------------
class SurpriseGate:
    """One-step prediction-error normaliser and surprise test.

    The gate itself is *stateless with respect to writes*; the system owns a
    single write cooldown shared by the surprise and novelty gates.
    """

    def __init__(self, cfg: MemoryConfig, warmup: int = 8):
        self.cfg = cfg
        self.warmup = int(warmup)
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.decay = float(cfg.surprise_ema)

    def reset(self) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, 1e-8))

    def raw_threshold(self) -> float:
        return self.mean + self.cfg.surprise_k * self.std

    def _z(self, raw_error: float) -> float:
        return max(0.0, (float(raw_error) - self.mean) / (self.std + 1e-6))

    def observe(self, raw_error: float) -> Tuple[float, bool]:
        """Update running statistics and return (surprise, is_surprising)."""
        raw_error = float(raw_error)
        self.count += 1
        if self.count == 1:
            self.mean = raw_error
            self.var = max(raw_error * raw_error, 1e-8)
        else:
            self.mean = self.decay * self.mean + (1 - self.decay) * raw_error
            self.var = self.decay * self.var + (1 - self.decay) * (raw_error - self.mean) ** 2
        z = self._z(raw_error)
        is_surprising = (
            self.count > self.warmup
            and raw_error > self.raw_threshold()
            and z > self.cfg.surprise_floor
        )
        return z, bool(is_surprising)

    def evaluate(self, raw_error: float) -> Tuple[float, bool]:
        """Frozen-statistics surprise test (eval mode, no updates)."""
        z = self._z(raw_error)
        is_surprising = (
            self.count > self.warmup
            and float(raw_error) > self.raw_threshold()
            and z > self.cfg.surprise_floor
        )
        return z, bool(is_surprising)

    def normalize(self, raw_error: float) -> float:
        return self._z(raw_error)

    def state_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: Dict[str, float]) -> None:
        self.mean = float(state["mean"])
        self.var = float(state["var"])
        self.count = int(state["count"])


# ---------------------------------------------------------------------------
# Working memory
# ---------------------------------------------------------------------------
class WorkingMemory:
    def __init__(self, cfg: MemoryConfig, dim: int, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.dim = int(dim)
        self.device = torch.device(device)
        self.slots = torch.zeros(cfg.wm_slots, dim, device=self.device)
        self.filled = torch.zeros(cfg.wm_slots, dtype=torch.bool, device=self.device)
        self.usage = torch.zeros(cfg.wm_slots, device=self.device)
        self.n_writes = 0
        self.last_slot = -1

    @property
    def n_filled(self) -> int:
        return int(self.filled.sum())

    @torch.no_grad()
    def read(self, query: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Content-based read. Returns (read_vector, best_similarity, entropy)."""
        if self.n_filled == 0:
            return torch.zeros(self.dim, device=self.device), 0.0, 0.0
        q = torch.nn.functional.normalize(query, dim=0)
        sim = self.slots @ q
        sim = sim.masked_fill(~self.filled, -2.0)
        attn = torch.softmax(self.cfg.wm_beta * sim, dim=0)
        read = attn @ self.slots
        best = float(sim.max())
        entropy = float(-(attn * torch.log(attn + 1e-9)).sum())
        self.usage.mul_(self.cfg.wm_usage_decay).add_(attn)
        return read, best, entropy

    @torch.no_grad()
    def write(self, content: torch.Tensor) -> int:
        c = torch.nn.functional.normalize(content, dim=0)
        if torch.isnan(c).any() or c.norm() < 1e-6:
            return -1
        if self.n_filled < self.cfg.wm_slots:
            slot = int(torch.nonzero(~self.filled)[0])
        else:
            slot = int(torch.argmin(self.usage))
        self.slots[slot] = c
        self.filled[slot] = True
        self.usage[slot] = 1.0
        self.n_writes += 1
        self.last_slot = slot
        return slot

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "slots": self.slots.cpu(),
            "filled": self.filled.cpu(),
            "usage": self.usage.cpu(),
            "n_writes": torch.tensor(self.n_writes),
        }

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.slots = state["slots"].to(self.device)
        self.filled = state["filled"].to(self.device)
        self.usage = state["usage"].to(self.device)
        self.n_writes = int(state["n_writes"].item())


# ---------------------------------------------------------------------------
# Attractor long-term memory
# ---------------------------------------------------------------------------
class AttractorMemory:
    def __init__(self, cfg: MemoryConfig, dim: int, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.dim = int(dim)
        self.device = torch.device(device)
        self.patterns = torch.zeros(cfg.attr_capacity, dim, device=self.device)
        self.count = 0          # number of valid prototypes (prefix is valid)
        self.ptr = 0            # ring-buffer write pointer
        self.n_writes = 0
        self.n_skipped = 0

    @torch.no_grad()
    def read(self, query: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
        if self.count == 0:
            return torch.zeros(self.dim, device=self.device), 0.0, -1
        q = torch.nn.functional.normalize(query, dim=0)
        sims = self.patterns[: self.count] @ q
        k = min(self.cfg.attr_topk, self.count)
        top = torch.topk(sims, k)
        weights = torch.softmax(self.cfg.attr_beta * top.values, dim=0)
        read = weights @ self.patterns[top.indices]
        return read, float(top.values[0]), int(top.indices[0])

    @torch.no_grad()
    def write(self, content: torch.Tensor, novelty_threshold: float | None = None) -> bool:
        c = torch.nn.functional.normalize(content, dim=0)
        if torch.isnan(c).any() or c.norm() < 1e-6:
            return False
        thr = self.cfg.novelty_cos if novelty_threshold is None else novelty_threshold
        if self.count > 0:
            max_sim = float((self.patterns[: self.count] @ c).max())
            if max_sim >= thr:
                self.n_skipped += 1
                return False
        self.patterns[self.ptr] = c
        self.ptr = (self.ptr + 1) % self.cfg.attr_capacity
        self.count = min(self.count + 1, self.cfg.attr_capacity)
        self.n_writes += 1
        return True

    @torch.no_grad()
    def retrieve_batch(self, queries: torch.Tensor) -> torch.Tensor:
        """Batched top-k attractor retrieval; queries [B, dim]."""
        if self.count == 0:
            return torch.zeros_like(queries)
        q = torch.nn.functional.normalize(queries, dim=-1)
        sims = q @ self.patterns[: self.count].T                 # [B, P]
        k = min(self.cfg.attr_topk, self.count)
        top = torch.topk(sims, k, dim=-1)
        w = torch.softmax(self.cfg.attr_beta * top.values, dim=-1)
        return (w.unsqueeze(-1) * self.patterns[top.indices]).sum(dim=-2)

    @torch.no_grad()
    def retrieval_accuracy(
        self, n_samples: int, noise: float, generator: torch.Generator | None = None
    ) -> Dict[str, float]:
        """Pattern completion test: corrupt a stored prototype and retrieve it.

        Returns mean cosine between retrieval and the clean prototype, plus a
        chance-level baseline (mean cosine to random other prototypes).
        """
        if self.count < 2:
            return {"cosine": 0.0, "chance_cosine": 0.0, "n": self.count}
        g = generator or torch.Generator(device="cpu")
        n = min(n_samples, self.count)
        idx = torch.randperm(self.count, generator=g)[:n]
        clean = self.patterns[idx]
        noisy = torch.nn.functional.normalize(
            clean + noise * torch.randn(clean.shape, generator=g), dim=-1
        ).to(self.device)
        read = self.retrieve_batch(noisy)
        cos = torch.nn.functional.cosine_similarity(read, clean, dim=-1).mean().item()
        # chance: average similarity of a retrieved pattern to a *different* stored one
        perm = torch.randperm(self.count, generator=g)
        other = self.patterns[perm[:n]]
        chance = torch.nn.functional.cosine_similarity(read, other, dim=-1).mean().item()
        return {"cosine": float(cos), "chance_cosine": float(chance), "n": int(n)}

    def stats(self) -> Dict[str, float]:
        if self.count == 0:
            return {"n_patterns": 0, "n_writes": self.n_writes, "n_skipped": self.n_skipped}
        p = torch.nn.functional.normalize(self.patterns[: self.count], dim=-1)
        sim = p @ p.T
        off = sim[~torch.eye(self.count, dtype=torch.bool, device=self.device)]
        return {
            "n_patterns": self.count,
            "n_writes": self.n_writes,
            "n_skipped": self.n_skipped,
            "mean_pairwise_cos": float(off.mean()) if off.numel() else 0.0,
        }

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "patterns": self.patterns.cpu(),
            "count": torch.tensor(self.count),
            "ptr": torch.tensor(self.ptr),
            "n_writes": torch.tensor(self.n_writes),
            "n_skipped": torch.tensor(self.n_skipped),
        }

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.patterns = state["patterns"].to(self.device)
        self.count = int(state["count"].item())
        self.ptr = int(state["ptr"].item())
        self.n_writes = int(state["n_writes"].item())
        self.n_skipped = int(state["n_skipped"].item())
