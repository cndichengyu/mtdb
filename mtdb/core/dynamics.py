"""Multi-timescale sparse recurrent dynamics (MTDB section A).

Per unit i (discrete update, dt = ``dt_ms``):

    x_i(t+1) = alpha_i * x_i(t) + (1 - alpha_i) * tanh(drive_i(t))
    drive_i  = sum_j W_ij * phi(x_j) + sum_k U_ik * s_k + b_i + noise
    alpha_i  = exp(-dt / tau_i)

Each unit belongs to one of three time-constant groups (fast / medium / slow).
Connectivity is sparse *by construction* (fixed fan-in ``in_degree``), so cost
scales O(N * K) and the same code can run a 100k-unit prototype.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from ..config import DynamicsConfig, PlasticityConfig
from .projection import SparseProjection


class MultiTimescaleDynamics:
    def __init__(
        self,
        cfg: DynamicsConfig,
        input_dim: int,
        generator: Optional[torch.Generator] = None,
        device: torch.device | str = "cpu",
    ):
        self.cfg = cfg
        self.input_dim = int(input_dim)
        self.device = torch.device(device)
        self.n_units = int(cfg.n_units)
        self.K = int(cfg.in_degree)
        # Use a local generator so that the core init is reproducible even if
        # the global RNG state has been consumed elsewhere.
        gen = generator or torch.Generator(device="cpu")
        if generator is None:
            gen.manual_seed(0)

        # --- sparse recurrent matrix (values + fixed pre-synaptic indices) --
        pre_idx = torch.randint(0, self.n_units, (self.n_units, self.K), generator=gen)
        w = torch.randn(self.n_units, self.K, generator=gen)
        sign = torch.where(
            torch.rand(self.n_units, self.K, generator=gen) < cfg.excitatory_fraction,
            torch.ones(()),
            -torch.ones(()),
        )
        w = w * sign / math.sqrt(self.K)
        self.pre_idx = pre_idx.long().to(self.device)
        self.W = w.to(self.device)
        self._rescale_to_spectral_radius(cfg.target_spectral_radius)

        # --- time constants --------------------------------------------------
        probs = torch.tensor(cfg.group_probs, dtype=torch.float32)
        probs = probs / probs.sum()
        groups = torch.multinomial(probs, self.n_units, replacement=True, generator=gen)
        taus = torch.tensor(cfg.tau_groups_ms, dtype=torch.float32)[groups]
        self.group_ids = groups.long().to(self.device)
        self.tau_ms = taus.to(self.device)
        self.decay = torch.exp(-cfg.dt_ms / self.tau_ms).to(self.device).clamp(0.0, 0.99999)

        # --- input, bias, state ---------------------------------------------
        self.W_in = (
            torch.randn(self.n_units, self.input_dim, generator=gen)
            / math.sqrt(self.input_dim)
        ).to(self.device)
        self.bias = torch.zeros(self.n_units, device=self.device)
        self.x = torch.zeros(self.n_units, device=self.device)
        self.rate = torch.full(
            (self.n_units,), float(cfg.target_rate), device=self.device
        )

        # --- fixed projection x -> h (interface latent) ----------------------
        self.h_proj = SparseProjection.random(
            self.n_units,
            cfg.h_dim,
            cfg.projection_fan_in,
            generator=gen,
            dtype=torch.float32,
        ).to(self.device)

        # --- reward / error traces for three-factor plasticity ---------------
        self.eligibility = torch.zeros_like(self.W)
        self.last_plasticity: Dict[str, float] = {}

    # ------------------------------------------------------------------ init
    def _apply_W(self, v: torch.Tensor) -> torch.Tensor:
        return (v[self.pre_idx] * self.W).sum(dim=-1)

    @torch.no_grad()
    def _rescale_to_spectral_radius(self, target: float, iters: int = 15) -> None:
        v = torch.randn(self.n_units, device=self.device)
        v = v / v.norm().clamp_min(1e-12)
        rho = 0.0
        for _ in range(iters):
            wv = self._apply_W(v)
            rho = wv.norm().item()
            if rho < 1e-12:
                break
            v = wv / wv.norm()
        if rho > 0:
            self.W.mul_(target / rho)

    # ----------------------------------------------------------------- state
    def reset(self) -> None:
        self.x.zero_()
        self.eligibility.zero_()

    @torch.no_grad()
    def project(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.x if x is None else x
        return self.h_proj(x)

    def phi(self) -> torch.Tensor:
        return torch.tanh(self.x)

    # ------------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, s: torch.Tensor, learn: bool = False) -> torch.Tensor:
        """Advance one tick. ``s`` is a [input_dim] tensor (no batch dim)."""
        cfg = self.cfg
        phix = torch.tanh(self.x)
        rec = (phix[self.pre_idx] * self.W).sum(dim=-1)
        drive = rec + self.W_in @ s + self.bias
        if cfg.noise_std > 0:
            drive = drive + cfg.noise_std * torch.randn(
                self.n_units, device=self.device
            )
        x_new = self.decay * self.x + (1.0 - self.decay) * torch.tanh(drive)
        self.x = x_new

        if learn:
            # Homeostatic bias: keep mean |x| near a target rate.
            self.rate.mul_(0.99).add_(0.01 * x_new.abs())
            self.bias.sub_((self.rate - cfg.target_rate) * cfg.homeostatic_lr)
            self.bias.clamp_(-cfg.bias_clip, cfg.bias_clip)
        return self.x

    # -------------------------------------------------------------- learning
    @torch.no_grad()
    def plastic_update(
        self,
        error_units: torch.Tensor,
        modulation: float = 1.0,
        reward: float = 0.0,
        pcfg: Optional[PlasticityConfig] = None,
    ) -> Dict[str, float]:
        """One step of three-factor local plasticity on the recurrent weights.

        delta W_ij =  lr_hebb * (post_i pre_j - post_i^2 W_ij)      [Oja]
                    + lr_error * modulation * e_i pre_j              [error]
                    + lr_reward * reward * eligibility_ij            [reward]
        """
        cfg = pcfg
        if cfg is None:
            return {}
        if not hasattr(self, "x") or self.x.numel() == 0:
            return {}
        phix = torch.tanh(self.x)
        pre = phix[self.pre_idx]                 # [N, K]
        post = phix                               # [N]
        post2 = (post * post).unsqueeze(1)

        # Eligibility trace: Hebbian coincidence with exponential memory.
        self.eligibility.mul_(cfg.eligibility_decay).add_(post.unsqueeze(1) * pre)

        hebb = post.unsqueeze(1) * pre - post2 * self.W
        err = (error_units.unsqueeze(1) * pre) * float(modulation)
        delta = cfg.lr_hebb * hebb + cfg.lr_error * err + cfg.lr_reward * float(reward) * self.eligibility
        delta = delta.clamp(-cfg.max_step, cfg.max_step)

        self.W.add_(delta)
        if cfg.weight_decay > 0:
            self.W.mul_(1.0 - cfg.weight_decay)
        self._cap_row_norm(cfg.row_norm_cap)

        self.last_plasticity = {
            "delta_abs_mean": float(delta.abs().mean()),
            "hebb_abs_mean": float(hebb.abs().mean()),
            "error_abs_mean": float(err.abs().mean()),
            "weight_abs_mean": float(self.W.abs().mean()),
            "eligibility_abs_mean": float(self.eligibility.abs().mean()),
        }
        return self.last_plasticity

    @torch.no_grad()
    def _cap_row_norm(self, cap: float) -> None:
        row = self.W.norm(dim=1).clamp_min(1e-8)
        scale = (cap / row).clamp(max=1.0)
        self.W.mul_(scale.unsqueeze(1))

    @torch.no_grad()
    def prune(self, prune_fraction: float) -> int:
        """Zero the weakest fraction of synapses per row (used by sleep)."""
        if prune_fraction <= 0:
            return 0
        k = max(1, int(round(self.K * prune_fraction)))
        weakest = self.W.abs().topk(k, dim=1, largest=False).indices
        before = int((self.W != 0).sum())
        self.W.scatter_(1, weakest, 0.0)
        after = int((self.W != 0).sum())
        return before - after

    @torch.no_grad()
    def normalize_rows(self) -> float:
        """Restore the pre-prune average row norm (homeostatic weight control)."""
        row = self.W.norm(dim=1).clamp_min(1e-8)
        scale = row.mean() / row
        self.W.mul_(scale.unsqueeze(1))
        return float(self.W.norm(dim=1).mean())

    # ------------------------------------------------------------------ stats
    @torch.no_grad()
    def stats(self) -> Dict[str, float]:
        return {
            "x_abs_mean": float(self.x.abs().mean()),
            "x_std": float(self.x.std()),
            "active_fraction": float((self.x.abs() > self.cfg.spike_threshold).float().mean()),
            "drive_saturated_fraction": float((self.x.abs() > 0.95).float().mean()),
            "weight_abs_mean": float(self.W.abs().mean()),
            "weight_zero_fraction": float((self.W == 0).float().mean()),
            "bias_abs_mean": float(self.bias.abs().mean()),
            "row_norm_mean": float(self.W.norm(dim=1).mean()),
        }

    # ------------------------------------------------------------ checkpoint
    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "W": self.W.detach().cpu(),
            "W_in": self.W_in.detach().cpu(),
            "bias": self.bias.detach().cpu(),
            "x": self.x.detach().cpu(),
            "rate": self.rate.detach().cpu(),
            "pre_idx": self.pre_idx.detach().cpu(),
            "group_ids": self.group_ids.detach().cpu(),
            "tau_ms": self.tau_ms.detach().cpu(),
            "eligibility": self.eligibility.detach().cpu(),
            "h_proj": self.h_proj.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.W = state["W"].to(self.device)
        self.W_in = state["W_in"].to(self.device)
        self.bias = state["bias"].to(self.device)
        self.x = state["x"].to(self.device)
        self.rate = state["rate"].to(self.device)
        self.pre_idx = state["pre_idx"].long().to(self.device)
        self.group_ids = state["group_ids"].long().to(self.device)
        self.tau_ms = state["tau_ms"].to(self.device)
        self.decay = torch.exp(-self.cfg.dt_ms / self.tau_ms).clamp(0.0, 0.99999)
        self.eligibility = state["eligibility"].to(self.device)
        self.h_proj = SparseProjection.from_state_dict(state["h_proj"]).to(self.device)
