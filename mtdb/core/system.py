"""MTDBSystem: the assembled multi-timescale dynamical base.

One ``step`` runs the full bottom layer:

    sensory input s(t)
      -> phase oscillators + sparse multi-timescale dynamics
      -> compressed latent h(t), phase-cluster features
      -> surprise = normalised one-step self-prediction error
      -> surprise-gated working-memory / attractor-memory read+write
      -> interface packet z(t) for the upper layer

Learning is split from inference: when ``learn=False`` the system is frozen
(no homeostatic adaptation, no self-predictor update, no memory writes, no
surprise-statistics update) which keeps evaluation and anomaly detection
well-defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from ..config import ExperimentConfig
from .dynamics import MultiTimescaleDynamics
from .interface import InterfacePacket, InterfaceSpec, pack_interface
from .memory import AttractorMemory, SurpriseGate, WorkingMemory
from .phase import PhaseSyncRouter
from .plasticity import ThreeFactorPlasticity
from .projection import SparseProjection
from ..training.readout import RLSReadout


@dataclass
class StepOutput:
    packet: InterfacePacket
    h: torch.Tensor
    base: torch.Tensor
    spike_fraction: float = 0.0
    surprise: float = 0.0
    phase_sync: float = 0.0
    gate: bool = False
    wm_write: bool = False
    attr_write: bool = False
    memory_key: int = -1
    metadata: Dict[str, float] = field(default_factory=dict)


class MTDBSystem:
    def __init__(
        self,
        cfg: ExperimentConfig,
        input_dim: int,
        upper_dim: int,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        self.cfg = cfg
        self.input_dim = int(input_dim)
        self.upper_dim = int(upper_dim)   # dimensionality of the readout target
        self.device = torch.device(device)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        self.gen = gen

        self.dynamics = MultiTimescaleDynamics(
            cfg.dynamics, input_dim, generator=gen, device=self.device
        )
        self.phase = PhaseSyncRouter(
            cfg.phase,
            n_units=cfg.dynamics.n_units,
            pre_idx=self.dynamics.pre_idx,
            group_ids=self.dynamics.group_ids,
            input_dim=input_dim,
            dt_ms=cfg.dynamics.dt_ms,
            generator=gen,
            device=self.device,
        )
        self.phase_dim = 3 * cfg.phase.n_clusters

        mcfg = cfg.memory
        self.wm: Optional[WorkingMemory] = None
        self.attractor: Optional[AttractorMemory] = None
        if mcfg.enabled:
            self.wm = WorkingMemory(mcfg, cfg.dynamics.h_dim, device=self.device)
            self.attractor = AttractorMemory(mcfg, cfg.dynamics.h_dim, device=self.device)
        self.wm_proj = SparseProjection.random(
            cfg.dynamics.h_dim, mcfg.wm_read_dim, 24, generator=gen
        ).to(self.device)
        self.attr_proj = SparseProjection.random(
            cfg.dynamics.h_dim, mcfg.attr_read_dim, 24, generator=gen
        ).to(self.device)

        self.spec = InterfaceSpec(
            h_dim=cfg.dynamics.h_dim,
            phase_dim=self.phase_dim,
            wm_dim=mcfg.wm_read_dim,
            attr_dim=mcfg.attr_read_dim,
            n_clusters=cfg.phase.n_clusters,
            input_dim=self.input_dim,
            include_sensory=bool(cfg.include_sensory_channel),
            include_sensory_trace=bool(cfg.include_sensory_trace),
        )
        self.self_pred = RLSReadout(
            in_dim=cfg.dynamics.h_dim + self.phase_dim,
            out_dim=cfg.dynamics.h_dim,
            # no forgetting for the self-model: it should accumulate statistics
            # rather than chase the latest states (which keeps surprise stable)
            forget=1.0,
            init_delta=1.0,
            theta_norm_cap=25.0,
        )
        self.gate = SurpriseGate(mcfg, warmup=8)
        self.plasticity = ThreeFactorPlasticity(
            cfg.plasticity, self.dynamics, error_dim=upper_dim, generator=gen
        )
        self.plasticity_enabled = bool(cfg.plasticity.enabled)

        self._prev_base: Optional[torch.Tensor] = None
        self._prev_pred: Optional[torch.Tensor] = None
        self._prev_s: Optional[torch.Tensor] = None
        self.sensory_trace: Optional[torch.Tensor] = (
            torch.zeros(self.input_dim, device=self.device)
            if cfg.include_sensory_trace else None
        )
        self._last: Optional[StepOutput] = None
        self._write_cooldown = 0
        self.step_count = 0

    # ------------------------------------------------------------------ reset
    def reset_state(self, randomize_phase: bool = True) -> None:
        self.dynamics.reset()
        if randomize_phase:
            self.phase.reset()
        self._prev_base = None
        self._prev_pred = None
        self._prev_s = None
        if self.sensory_trace is not None:
            self.sensory_trace.zero_()

    # --------------------------------------------------------- state snapshots
    def snapshot_state(self, light: bool = False) -> Dict[str, object]:
        """Cheap snapshot used by active inference / mental rollouts.

        Memory contents are included because even a read-only simulated step can
        change working-memory usage statistics. Restoring a snapshot must leave
        the real system bit-identical.
        """
        snap: Dict[str, object] = {
            "x": self.dynamics.x.detach().clone(),
            "theta": self.phase.theta.detach().clone(),
            "prev_s": self._prev_s.detach().clone() if self._prev_s is not None else None,
            "prev_base": (
                self._prev_base.detach().clone() if self._prev_base is not None else None
            ),
            "prev_pred": (
                self._prev_pred.detach().clone() if self._prev_pred is not None else None
            ),
            "write_cooldown": int(self._write_cooldown),
            "step_count": int(self.step_count),
            "sensory_trace": (
                self.sensory_trace.detach().clone()
                if self.sensory_trace is not None else None
            ),
        }
        if self.wm is not None:
            if light:
                # read-only rollouts only mutate the usage statistics
                snap["wm_usage"] = self.wm.usage.detach().clone()
            else:
                snap["wm"] = {k: v.clone() for k, v in self.wm.state_dict().items()}
        if self.attractor is not None and not light:
            snap["attractor"] = {
                k: v.clone() for k, v in self.attractor.state_dict().items()
            }
        return snap

    def restore_state(self, snap: Dict[str, object]) -> None:
        self.dynamics.x = snap["x"].clone()
        self.phase.theta = snap["theta"].clone()
        self._prev_s = (
            snap["prev_s"].clone() if snap["prev_s"] is not None else None
        )
        self._prev_base = (
            snap["prev_base"].clone() if snap["prev_base"] is not None else None
        )
        self._prev_pred = (
            snap["prev_pred"].clone() if snap["prev_pred"] is not None else None
        )
        self._write_cooldown = int(snap["write_cooldown"])
        self.step_count = int(snap["step_count"])
        if snap["sensory_trace"] is not None and self.sensory_trace is not None:
            self.sensory_trace = snap["sensory_trace"].clone()
        if self.wm is not None:
            if "wm" in snap:
                self.wm.load_state_dict(snap["wm"])
            elif "wm_usage" in snap:
                self.wm.usage = snap["wm_usage"].detach().clone()
        if self.attractor is not None and "attractor" in snap:
            self.attractor.load_state_dict(snap["attractor"])

    # ------------------------------------------------------------------- step
    @torch.no_grad()
    def step(
        self,
        s: torch.Tensor,
        learn: bool = False,
        allow_memory_write: Optional[bool] = None,
    ) -> StepOutput:
        s = s.to(self.device, dtype=torch.float32)
        self.dynamics.step(s, learn=learn)
        x = self.dynamics.x

        if self.cfg.phase.enabled:
            self.phase.step(s)
            phase_feat, sync, cluster_ids = self.phase.features(x)
        else:
            phase_feat = torch.zeros(self.phase_dim, device=self.device)
            sync = 0.0
            cluster_ids = torch.zeros(self.dynamics.n_units, dtype=torch.long,
                                     device=self.device)

        h = self.dynamics.project(x)
        base = torch.cat([h, phase_feat], dim=0)

        if allow_memory_write is None:
            allow_memory_write = learn

        # ---- surprise: one-step self-prediction error --------------------
        surprise = 0.0
        surprise_gate = False
        if self._prev_pred is not None:
            raw = torch.sqrt(torch.mean((h - self._prev_pred) ** 2) + 1e-12).item()
            if learn:
                surprise, surprise_gate = self.gate.observe(raw)
                if self._prev_base is not None:
                    self.self_pred.update(self._prev_base, h)
            else:
                surprise, surprise_gate = self.gate.evaluate(raw)
        self._prev_pred = self.self_pred.predict(base)
        self._prev_base = base

        # ---- memory reads ------------------------------------------------
        wm_read = torch.zeros(self.cfg.dynamics.h_dim, device=self.device)
        attr_read = torch.zeros(self.cfg.dynamics.h_dim, device=self.device)
        wm_hit = 0.0
        attr_hit = 0.0
        wm_write = False
        attr_write = False
        memory_key = -1
        if self.wm is not None:
            wm_read, wm_hit, _ = self.wm.read(h)
        if self.attractor is not None:
            attr_read, attr_hit, memory_key = self.attractor.read(h)

        # ---- write gate: prediction surprise OR content novelty ----------
        # Novelty is the "surprise of the memory system": content that no
        # existing slot/prototype explains. Both signals are surprise about
        # the world; combining them keeps memory useful when the self-model is
        # already good but the situation is new.
        novelty = False
        if allow_memory_write:
            mcfg = self.cfg.memory
            if self.wm is not None and wm_hit < mcfg.wm_novelty_threshold:
                novelty = True
            if self.attractor is not None and attr_hit < mcfg.attr_novelty_threshold:
                novelty = True
        gate = bool(
            allow_memory_write
            and self._write_cooldown == 0
            and (surprise_gate or novelty)
        )
        if gate:
            if self.wm is not None:
                wm_write = self.wm.write(h) >= 0
            if self.attractor is not None:
                attr_write = self.attractor.write(h)
                if attr_write:
                    memory_key = (self.attractor.ptr - 1) % self.attractor.cfg.attr_capacity
            self._write_cooldown = int(self.cfg.memory.write_cooldown)
        elif self._write_cooldown > 0:
            self._write_cooldown -= 1


        wm_proj = self.wm_proj(wm_read) if self.wm is not None else torch.zeros(
            self.cfg.memory.wm_read_dim, device=self.device
        )
        attr_proj = self.attr_proj(attr_read) if self.attractor is not None else torch.zeros(
            self.cfg.memory.attr_read_dim, device=self.device
        )
        confidence = 1.0 / (1.0 + surprise)

        if self.sensory_trace is not None:
            decay = float(self.cfg.memory.sensory_trace_decay)
            self.sensory_trace.mul_(decay).add_(s)

        z = pack_interface(
            self.spec, h, phase_feat, wm_proj, attr_proj,
            surprise=surprise, sync=sync, confidence=confidence,
            s=s, prev_s=self._prev_s, trace=self.sensory_trace,
        )
        self._prev_s = s.detach().clone()
        spike_fraction = float(
            (x.abs() > self.cfg.dynamics.spike_threshold).float().mean()
        )
        packet = InterfacePacket(
            z=z,
            h_slow=h,
            phase_features=phase_feat,
            wm_read=wm_proj,
            attractor_read=attr_proj,
            surprise=surprise,
            sync=sync,
            confidence=confidence,
            cluster_membership=cluster_ids,
            sensory_trace=(
                self.sensory_trace.detach().clone()
                if self.sensory_trace is not None else None
            ),
            memory_key=memory_key if memory_key >= 0 else None,
            wrote_memory=wm_write or attr_write,
            metadata={
                "wm_hit": wm_hit,
                "attr_hit": attr_hit,
                "spike_fraction": spike_fraction,
                "wm_filled": float(self.wm.n_filled if self.wm is not None else 0),
                "attr_count": float(self.attractor.count if self.attractor is not None else 0),
                "novelty": float(novelty),
                "surprise_gate": float(surprise_gate),
            },
        )
        out = StepOutput(
            packet=packet,
            h=h,
            base=base,
            spike_fraction=spike_fraction,
            surprise=surprise,
            phase_sync=sync,
            gate=gate,
            wm_write=wm_write,
            attr_write=attr_write,
            memory_key=memory_key,
            metadata=packet.metadata,
        )
        self._last = out
        self.step_count += 1
        return out

    # -------------------------------------------------------------- plasticity
    @torch.no_grad()
    def apply_error_feedback(
        self,
        error: torch.Tensor,
        surprise: float = 0.0,
        reward: float = 0.0,
        attention: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        if not self.plasticity_enabled:
            return {}
        return self.plasticity.step(
            error.to(self.device), surprise=surprise, reward=reward, attention=attention
        )

    # ------------------------------------------------------------------ replay
    @torch.no_grad()
    def replay(
        self, inputs: torch.Tensor, allow_memory_write: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Run a stored input window through a fresh state (used by sleep)."""
        self.reset_state()
        zs: List[torch.Tensor] = []
        hs: List[torch.Tensor] = []
        surprises: List[float] = []
        ctx: List[torch.Tensor] = []
        for t in range(inputs.shape[0]):
            out = self.step(
                inputs[t], learn=False, allow_memory_write=allow_memory_write
            )
            zs.append(out.packet.z)
            hs.append(out.h)
            surprises.append(out.surprise)
            ctx.append(out.base)
        return {
            "z": torch.stack(zs) if zs else torch.zeros(0, self.spec.total_dim),
            "h": torch.stack(hs) if hs else torch.zeros(0, self.cfg.dynamics.h_dim),
            "surprise": torch.tensor(surprises),
            "base": torch.stack(ctx) if ctx else torch.zeros(0, self.cfg.dynamics.h_dim + self.phase_dim),
        }

    # ------------------------------------------------------------------ stats
    def stats(self) -> Dict[str, float]:
        out = self.dynamics.stats()
        out["step_count"] = float(self.step_count)
        if self.wm is not None:
            out["wm_filled"] = float(self.wm.n_filled)
            out["wm_writes"] = float(self.wm.n_writes)
        if self.attractor is not None:
            out.update({f"attr_{k}": float(v) for k, v in self.attractor.stats().items()})
        out["interface_dim"] = float(self.spec.total_dim)
        out["surprise_mean"] = float(self.gate.mean)
        out["surprise_std"] = float(self.gate.std)
        return out

    # ------------------------------------------------------------ checkpoint
    def state_dict(self) -> Dict[str, object]:
        state: Dict[str, object] = {
            "dynamics": self.dynamics.state_dict(),
            "phase": self.phase.state_dict(),
            "self_pred": self.self_pred.state_dict(),
            "gate": self.gate.state_dict(),
            "plasticity_feedback": self.plasticity.feedback.state_dict(),
            "wm_proj": self.wm_proj.state_dict(),
            "attr_proj": self.attr_proj.state_dict(),
            "config": self.cfg.to_dict(),
            "input_dim": self.input_dim,
            "upper_dim": self.upper_dim,
            "step_count": self.step_count,
            "write_cooldown": self._write_cooldown,
            "sensory_trace": (
                self.sensory_trace.detach().cpu()
                if self.sensory_trace is not None else None
            ),
        }
        if self.wm is not None:
            state["wm"] = self.wm.state_dict()
        if self.attractor is not None:
            state["attractor"] = self.attractor.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.dynamics.load_state_dict(state["dynamics"])
        self.phase.load_state_dict(state["phase"])
        self.self_pred.load_state_dict(state["self_pred"])
        self.gate.load_state_dict(state["gate"])
        self.plasticity.feedback.load_state_dict(state["plasticity_feedback"])
        self.wm_proj = SparseProjection.from_state_dict(state["wm_proj"]).to(self.device)
        self.attr_proj = SparseProjection.from_state_dict(state["attr_proj"]).to(self.device)
        self.step_count = int(state.get("step_count", 0))
        self._write_cooldown = int(state.get("write_cooldown", 0))
        if state.get("sensory_trace") is not None and self.sensory_trace is not None:
            self.sensory_trace = state["sensory_trace"].to(self.device)
        if self.wm is not None and "wm" in state:
            self.wm.load_state_dict(state["wm"])
        if self.attractor is not None and "attractor" in state:
            self.attractor.load_state_dict(state["attractor"])

    def n_parameters(self) -> int:
        total = int(self.dynamics.W.numel() + self.dynamics.W_in.numel())
        total += int(self.dynamics.h_proj.vals.numel())
        total += int(self.phase.phase_W_in.numel())
        total += int(self.self_pred.theta.numel())
        total += int(self.wm_proj.vals.numel() + self.attr_proj.vals.numel())
        if self.wm is not None:
            total += int(self.wm.slots.numel())
        if self.attractor is not None:
            total += int(self.attractor.patterns.numel())
        return total
