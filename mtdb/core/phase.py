"""Phase synchronization and dynamic routing (MTDB section B).

Each unit carries a Kuramoto phase:

    dtheta_i/dt = 2*pi*f_i + K_c * mean_j sin(theta_j - theta_i)
                  + K_in * (U_phase s)_i + noise

Units whose phases fall in the same bin form a transient synchronization
cluster. Clusters are the base's mechanism for feature binding and for routing
information to memory and to the interface. The emitted phase features are, for
each cluster m:

    [ r_m, mean(x)_m, fraction_m ]   with r_m = |mean exp(i theta_m)|

so ``3 * n_clusters`` dimensions in total. This gives the upper layer an
explicit, interpretable "what is bound together right now" signal instead of
hiding it inside a feed-forward embedding.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from ..config import PhaseConfig


class PhaseSyncRouter:
    def __init__(
        self,
        cfg: PhaseConfig,
        n_units: int,
        pre_idx: torch.Tensor,
        group_ids: torch.Tensor,
        input_dim: int,
        dt_ms: float,
        generator: Optional[torch.Generator] = None,
        device: torch.device | str = "cpu",
    ):
        self.cfg = cfg
        self.n_units = int(n_units)
        self.n_clusters = int(cfg.n_clusters)
        self.K = int(pre_idx.shape[1])
        self.device = torch.device(device)
        gen = generator or torch.Generator(device="cpu")
        if generator is None:
            gen.manual_seed(1)

        self.dt_s = float(dt_ms) / 1000.0
        self.pre_idx = pre_idx.long().to(self.device)

        # Base frequency per time-constant group + per-unit jitter.
        freq_table = torch.tensor(cfg.freq_hz_by_group, dtype=torch.float32)
        freq = freq_table[group_ids.long().cpu()].to(self.device)
        jitter = cfg.freq_jitter_hz * torch.randn(self.n_units, generator=gen)
        self.freq_hz = (freq + jitter.to(self.device)).clamp_min(0.5)
        self.omega = 2.0 * math.pi * self.freq_hz

        # Input-dependent phase reset: same sensory event nudges co-active units.
        self.phase_W_in = (
            torch.randn(self.n_units, input_dim, generator=gen) / math.sqrt(input_dim)
        ).to(self.device)

        self.theta = (2.0 * math.pi * torch.rand(self.n_units, generator=gen)).to(self.device)

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.theta.uniform_(0.0, 2.0 * math.pi)

    @property
    def coupling_per_synapse(self) -> float:
        return self.cfg.coupling_gain / max(1, self.K)

    @torch.no_grad()
    def step(self, s: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        coupling = torch.sin(
            self.theta[self.pre_idx] - self.theta.unsqueeze(1)
        ).mean(dim=-1)
        drive = (
            self.freq_hz
            + cfg.coupling_gain * coupling
            + cfg.input_gain * (self.phase_W_in @ s)
        )
        dtheta = 2.0 * math.pi * self.dt_s * drive
        if cfg.phase_noise > 0:
            dtheta = dtheta + cfg.phase_noise * torch.randn(
                self.n_units, device=self.device
            )
        self.theta = torch.remainder(self.theta + dtheta, 2.0 * math.pi)
        return self.theta

    # --------------------------------------------------------------- features
    @torch.no_grad()
    def features(self, x: torch.Tensor) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """Return (phase_features [3C], global_sync scalar, cluster_ids [N])."""
        C = self.n_clusters
        bins = (self.theta / (2.0 * math.pi) * C).long().clamp_(0, C - 1)
        cos_t = torch.cos(self.theta)
        sin_t = torch.sin(self.theta)

        count = torch.bincount(bins, minlength=C).to(self.theta.dtype).clamp_min(1.0)
        sum_cos = torch.zeros(C, device=self.device).index_add_(0, bins, cos_t)
        sum_sin = torch.zeros(C, device=self.device).index_add_(0, bins, sin_t)
        sum_x = torch.zeros(C, device=self.device).index_add_(0, bins, x)
        coherence = torch.sqrt(sum_cos.pow(2) + sum_sin.pow(2)) / count
        mean_x = sum_x / count
        frac = count / float(self.n_units)

        # [C, 3] -> flattened as r_0..r_{C-1}, mx_0..mx_{C-1}, frac_0..frac_{C-1}
        feat = torch.stack([coherence, mean_x, frac], dim=1).reshape(-1)
        sync = float((coherence * frac).sum().clamp(0.0, 1.0))
        return feat, sync, bins

    # ------------------------------------------------------------- checkpoint
    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "theta": self.theta.detach().cpu(),
            "freq_hz": self.freq_hz.detach().cpu(),
            "omega": self.omega.detach().cpu(),
            "phase_W_in": self.phase_W_in.detach().cpu(),
        }

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.theta = state["theta"].to(self.device)
        self.freq_hz = state["freq_hz"].to(self.device)
        self.omega = state["omega"].to(self.device)
        self.phase_W_in = state["phase_W_in"].to(self.device)
