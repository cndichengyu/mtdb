"""Fixed sparse random projections used to compress unit states.

These are intentionally *not* trained: they are the read-out structure through
which a large dynamical core is exposed to a small interface. Sparse gather
keeps the cost O(n_out * fan_in) instead of O(n_out * n_in).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SparseProjection:
    idx: torch.Tensor      # [n_out, fan_in] int64
    vals: torch.Tensor     # [n_out, fan_in] float
    n_in: int
    n_out: int
    fan_in: int

    @classmethod
    def random(
        cls,
        n_in: int,
        n_out: int,
        fan_in: int,
        generator: Optional[torch.Generator] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "SparseProjection":
        fan_in = min(fan_in, n_in)
        idx = torch.randint(0, n_in, (n_out, fan_in), generator=generator)
        vals = torch.randn(n_out, fan_in, generator=generator) / (fan_in ** 0.5)
        return cls(
            idx=idx.long(),
            vals=vals.to(dtype),
            n_in=n_in,
            n_out=n_out,
            fan_in=fan_in,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x[self.idx] * self.vals).sum(dim=-1)

    def to(self, device: torch.device) -> "SparseProjection":
        self.idx = self.idx.to(device)
        self.vals = self.vals.to(device)
        return self

    def state_dict(self) -> dict:
        return {
            "idx": self.idx.cpu(),
            "vals": self.vals.cpu(),
            "meta": (self.n_in, self.n_out, self.fan_in),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "SparseProjection":
        n_in, n_out, fan_in = state["meta"]
        return cls(
            idx=state["idx"].long(),
            vals=state["vals"],
            n_in=int(n_in),
            n_out=int(n_out),
            fan_in=int(fan_in),
        )
