"""Three-factor local plasticity (MTDB section D).

The upper layer broadcasts a prediction error e(t) and a scalar reward r(t).
A fixed random feedback matrix B maps the error to units:

    e_units = B e(t)

which is the feedback-alignment style "global error broadcast" — no symmetric
backward pass and no backpropagation through time. The recurrent weight update
itself is fully local: it only needs pre-synaptic activity, post-synaptic
activity/error and a modulation scalar.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from ..config import PlasticityConfig
from .dynamics import MultiTimescaleDynamics


class ErrorFeedback:
    """Fixed random feedback matrix mapping feature-space error to unit space."""

    def __init__(
        self,
        error_dim: int,
        n_units: int,
        generator: Optional[torch.Generator] = None,
        device: torch.device | str = "cpu",
    ):
        gen = generator or torch.Generator(device="cpu")
        if generator is None:
            gen.manual_seed(2)
        self.B = (
            torch.randn(error_dim, n_units, generator=gen) / math.sqrt(error_dim)
        ).to(device)
        self.device = torch.device(device)

    def __call__(self, error: torch.Tensor) -> torch.Tensor:
        return error @ self.B

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"B": self.B.cpu()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.B = state["B"].to(self.device)


class ThreeFactorPlasticity:
    def __init__(
        self,
        cfg: PlasticityConfig,
        dynamics: MultiTimescaleDynamics,
        error_dim: int,
        generator: Optional[torch.Generator] = None,
    ):
        self.cfg = cfg
        self.dynamics = dynamics
        self.feedback = ErrorFeedback(
            error_dim,
            dynamics.n_units,
            generator=generator,
            device=dynamics.device,
        )
        self.steps = 0

    @torch.no_grad()
    def step(
        self,
        error: torch.Tensor,
        surprise: float,
        reward: float = 0.0,
        attention: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        if not self.cfg.enabled:
            return {}
        modulation = 1.0 + max(0.0, float(surprise)) * self.cfg.modulation_gain
        e_units = self.feedback(error)
        if attention is not None:
            e_units = e_units * attention
        stats = self.dynamics.plastic_update(
            e_units, modulation=modulation, reward=float(reward), pcfg=self.cfg
        )
        self.steps += 1
        stats = dict(stats)
        stats["modulation"] = float(modulation)
        stats["reward"] = float(reward)
        return stats

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.feedback.state_dict()

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.feedback.load_state_dict(state)
