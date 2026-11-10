"""Upper-layer interface protocol z(t) (MTDB section 4).

The base never emits tokens; it emits a compressed packet:

    z(t) = [ h_slow | Phi | M_wm | M_ltm | (surprise, sync, confidence) ]

``InterfaceSpec`` owns the slicing layout so downstream world models, symbolic
reasoners and agent policies can address fields by name. ``pack_interface`` is
the single place where the numerical layout is defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class InterfaceSpec:
    h_dim: int
    phase_dim: int
    wm_dim: int
    attr_dim: int
    n_clusters: int
    input_dim: int = 0
    include_sensory: bool = True
    include_sensory_trace: bool = False

    @property
    def aux_dim(self) -> int:
        return 3

    @property
    def sensory_dim(self) -> int:
        return 2 * self.input_dim if (self.include_sensory and self.input_dim > 0) else 0

    @property
    def trace_dim(self) -> int:
        return self.input_dim if (self.include_sensory_trace and self.input_dim > 0) else 0

    @property
    def total_dim(self) -> int:
        return (
            self.h_dim + self.phase_dim + self.wm_dim + self.attr_dim
            + self.sensory_dim + self.trace_dim + self.aux_dim
        )

    def slices(self) -> Dict[str, slice]:
        a = 0
        out = {
            "h": slice(a, a + self.h_dim),
            "phase": slice(a + self.h_dim, a + self.h_dim + self.phase_dim),
            "wm": slice(a + self.h_dim + self.phase_dim,
                        a + self.h_dim + self.phase_dim + self.wm_dim),
        }
        a = a + self.h_dim + self.phase_dim + self.wm_dim
        out["attractor"] = slice(a, a + self.attr_dim)
        a += self.attr_dim
        if self.sensory_dim > 0:
            out["input"] = slice(a, a + self.input_dim)
            a += self.input_dim
            out["delta"] = slice(a, a + self.input_dim)
            a += self.input_dim
        if self.trace_dim > 0:
            out["trace"] = slice(a, a + self.trace_dim)
            a += self.trace_dim
        out["aux"] = slice(a, a + self.aux_dim)
        return out


@dataclass
class InterfacePacket:
    """Semantic view of one interface tick. ``z`` is the concatenated vector."""

    z: torch.Tensor
    h_slow: torch.Tensor
    phase_features: torch.Tensor
    wm_read: torch.Tensor
    attractor_read: torch.Tensor
    surprise: float = 0.0
    sync: float = 0.0
    confidence: float = 0.0
    cluster_membership: Optional[torch.Tensor] = None
    sensory_trace: Optional[torch.Tensor] = None
    memory_key: Optional[int] = None
    wrote_memory: bool = False
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class UpperModulation:
    """Signals the upper layer sends back down into the base."""

    reward: float = 0.0
    attention: Optional[torch.Tensor] = None   # per-unit mask, optional
    query: Optional[torch.Tensor] = None       # retrieval key override
    target_error: Optional[torch.Tensor] = None  # [F] prediction error broadcast


def pack_interface(
    spec: InterfaceSpec,
    h: torch.Tensor,
    phase_features: torch.Tensor,
    wm_read: torch.Tensor,
    attractor_read: torch.Tensor,
    surprise: float,
    sync: float,
    confidence: float,
    s: Optional[torch.Tensor] = None,
    prev_s: Optional[torch.Tensor] = None,
    trace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Assemble the compressed interface vector. Dimensions are validated."""
    if h.shape[-1] != spec.h_dim:
        raise ValueError(f"h_dim mismatch: {h.shape[-1]} != {spec.h_dim}")
    if phase_features.shape[-1] != spec.phase_dim:
        raise ValueError(
            f"phase_dim mismatch: {phase_features.shape[-1]} != {spec.phase_dim}"
        )
    if wm_read.shape[-1] != spec.wm_dim:
        raise ValueError(f"wm_dim mismatch: {wm_read.shape[-1]} != {spec.wm_dim}")
    if attractor_read.shape[-1] != spec.attr_dim:
        raise ValueError(
            f"attr_dim mismatch: {attractor_read.shape[-1]} != {spec.attr_dim}"
        )
    parts = [h, phase_features, wm_read, attractor_read]
    if spec.sensory_dim > 0:
        if s is None:
            raise ValueError("sensory channel enabled but s was not provided")
        if s.shape[-1] != spec.input_dim:
            raise ValueError(
                f"input_dim mismatch: {s.shape[-1]} != {spec.input_dim}"
            )
        delta = torch.zeros_like(s) if prev_s is None else (s - prev_s)
        parts.extend([s, delta])
    if spec.trace_dim > 0:
        if trace is None:
            raise ValueError("sensory trace enabled but trace was not provided")
        if trace.shape[-1] != spec.trace_dim:
            raise ValueError(
                f"trace_dim mismatch: {trace.shape[-1]} != {spec.trace_dim}"
            )
        parts.append(trace)
    aux = torch.tensor([surprise, sync, confidence], dtype=h.dtype, device=h.device)
    parts.append(aux)
    return torch.cat(parts, dim=-1)
