"""Configuration objects for the MTDB training base.

All hyper-parameters live in nested dataclasses so they are typed, documentable
and introspectable. YAML files in ``configs/`` are loaded with
``ExperimentConfig.from_yaml``. Dotted overrides such as
``dynamics.n_units=2048`` are supported via ``apply_overrides``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Tuple, get_origin, get_type_hints

import yaml


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
@dataclass
class DynamicsConfig:
    """Multi-timescale sparse recurrent core (see MTDB section A)."""

    n_units: int = 1024
    in_degree: int = 16                 # pre-synaptic fan-in per unit (~1.5% at N=1024)
    tau_groups_ms: Tuple[float, ...] = (10.0, 100.0, 1000.0)
    group_probs: Tuple[float, ...] = (0.5, 0.3, 0.2)
    dt_ms: float = 10.0
    target_spectral_radius: float = 1.0
    noise_std: float = 0.01
    activation: str = "tanh"            # kept for extension; tanh implemented
    homeostatic_lr: float = 0.002
    target_rate: float = 0.12           # target mean |x|
    bias_clip: float = 2.0
    spike_threshold: float = 0.25       # |x| > threshold counted as an event
    row_norm_cap: float = 2.0
    h_dim: int = 96                     # compressed latent width of x
    projection_fan_in: int = 24         # sparse fixed random projection fan-in
    excitatory_fraction: float = 0.8


@dataclass
class PhaseConfig:
    """Kuramoto phase synchronization / dynamic routing (section B)."""

    enabled: bool = True
    freq_hz_by_group: Tuple[float, ...] = (8.0, 6.0, 4.0)
    freq_jitter_hz: float = 0.5
    coupling_gain: float = 0.4
    input_gain: float = 1.0
    phase_noise: float = 0.05
    n_clusters: int = 8
    sync_floor: float = 0.15            # min weighted coherence treated as "routed"


@dataclass
class MemoryConfig:
    """Working memory + attractor long-term memory (section C)."""

    enabled: bool = True
    wm_slots: int = 32
    wm_read_dim: int = 32               # compressed WM readout fed to interface
    wm_beta: float = 12.0
    wm_usage_decay: float = 0.999
    attr_capacity: int = 512            # number of attractor prototypes
    attr_read_dim: int = 32
    attr_topk: int = 8
    attr_beta: float = 8.0
    novelty_cos: float = 0.92           # skip writes too similar to stored prototype
    wm_novelty_threshold: float = 0.55  # write WM when content is this novel
    attr_novelty_threshold: float = 0.65  # write LTM when no prototype this close
    sensory_trace_decay: float = 0.8    # decay of the optional sensory evidence trace
    surprise_k: float = 2.0             # write when surprise > mean + k * std
    surprise_floor: float = 1.0
    surprise_ema: float = 0.99
    write_cooldown: int = 2
    write_during_training: bool = True


@dataclass
class PlasticityConfig:
    """Three-factor local plasticity (section D). No backpropagation."""

    enabled: bool = True
    lr_hebb: float = 1.0e-4
    lr_error: float = 4.0e-4
    lr_reward: float = 1.0e-4
    eligibility_decay: float = 0.95
    weight_decay: float = 1.0e-6
    max_step: float = 5.0e-3
    row_norm_cap: float = 2.0
    modulation_gain: float = 1.0


@dataclass
class ConsolidationConfig:
    """Sleep replay / consolidation (section E)."""

    enabled: bool = True
    buffer_size: int = 64
    window_len: int = 48
    replay_passes: int = 2
    prune_fraction: float = 0.01        # fraction of weakest synapses per row removed
    renorm: bool = True
    retrieval_samples: int = 32
    retrieval_noise: float = 0.35
    readout_refit: bool = True          # replay based least-squares readout consolidation


@dataclass
class ReadoutConfig:
    """Upper-layer readout trained without backpropagation."""

    kind: str = "ridge"                 # "ridge" (batch least squares) | "rls" (online)
    ridge_lambda: float = 0.05
    rls_forget: float = 0.9999
    rls_init_delta: float = 100.0


@dataclass
class TrainConfig:
    seq_len: int = 96
    n_train: int = 64
    n_eval: int = 16
    epochs: int = 3                     # stage-1 collection / refit passes
    horizon: int = 5                    # predict s_{t+horizon}
    log_every: int = 1
    online_epochs: int = 2              # stage-2 online passes over regime B
    online_steps: int = 0               # 0 -> whole stream, else truncate
    replay_stride: int = 32             # steps between replay windows while training
    run_stage2: bool = True
    run_recall: bool = True
    n_recall_train: int = 96
    n_recall_eval: int = 24
    forgetting_stress: bool = True
    stress_rls_forget: float = 0.999
    stress_online_epochs: int = 2
    ablations: bool = True
    ablation_n_train: int = 32          # smaller train set for ablation variants
    ablation_epochs: int = 2
    plot: bool = True
    save_model: bool = True
    save_readout: bool = True


@dataclass
class TaskConfig:
    prediction: str = "multimodal"      # name of the synthetic prediction world
    anomaly_rate: float = 0.02
    anomaly_window: int = 2
    recall_classes: int = 8
    recall_delay: int = 40
    recall_eval_delays: Tuple[int, ...] = (24, 40, 80, 120)
    recall_cue_len: int = 3
    recall_distractors: int = 3
    seed_train: int = 1234
    seed_eval: int = 5678


@dataclass
class ExperimentConfig:
    seed: int = 0
    device: str = "cpu"
    out_dir: str = "artifacts"
    run_name: str = "demo"
    include_sensory_channel: bool = True
    include_sensory_trace: bool = False
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    phase: PhaseConfig = field(default_factory=PhaseConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    plasticity: PlasticityConfig = field(default_factory=PlasticityConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    task: TaskConfig = field(default_factory=TaskConfig)

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        return _build(cls, data or {})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    # -- introspection -----------------------------------------------------
    def n_interface_dims(self, input_dim: int = 10) -> int:
        d = self.dynamics
        p = self.phase
        m = self.memory
        sensory = 2 * int(input_dim) if self.include_sensory_channel else 0
        trace = int(input_dim) if self.include_sensory_trace else 0
        return d.h_dim + 3 * p.n_clusters + m.wm_read_dim + m.attr_read_dim + sensory + trace + 3

    def fingerprint(self) -> str:
        parts = [
            f"N{self.dynamics.n_units}",
            f"K{self.dynamics.in_degree}",
            f"tau{'-'.join(str(int(t)) for t in self.dynamics.tau_groups_ms)}",
            f"ph{int(self.phase.enabled)}",
            f"mem{int(self.memory.enabled)}",
            f"pl{int(self.plasticity.enabled)}",
        ]
        return "_".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build(cls: type, data: Dict[str, Any]):
    """Recursively build a dataclass from a plain dict."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict for {cls.__name__}, got {type(data)}")
    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        kwargs[f.name] = _coerce(data[f.name], hints.get(f.name, f.type))
    return cls(**kwargs)


def _coerce(value: Any, hint: Any) -> Any:
    if value is None:
        return value
    if dataclasses.is_dataclass(hint):
        return _build(hint, value)
    origin = get_origin(hint)
    if origin is tuple and isinstance(value, (list, tuple)):
        return tuple(value)
    if origin in (list, List) and isinstance(value, tuple):
        return list(value)
    if hint is bool and isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return value


def apply_overrides(cfg: ExperimentConfig, overrides: List[str]) -> ExperimentConfig:
    """Apply ``a.b.c=value`` overrides parsed as YAML scalars."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        cursor: Any = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if not hasattr(cursor, part):
                raise AttributeError(f"Unknown config key section: {part!r} in {item!r}")
            cursor = getattr(cursor, part)
        leaf = parts[-1]
        if not hasattr(cursor, leaf):
            raise AttributeError(f"Unknown config key: {key!r}")
        setattr(cursor, leaf, value)
    return cfg


# Convenience: the three shipped presets are small enough to run on CPU.
PRESETS = {
    "debug": "configs/debug.yaml",
    "standard": "configs/standard.yaml",
    "mvp": "configs/mvp.yaml",
}
