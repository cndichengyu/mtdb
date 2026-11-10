"""Sleep replay and consolidation (MTDB section E).

Three jobs are performed offline:

1. Replay stored trajectories through the base (no plasticity).
2. Abstract recurring high-surprise states into attractor prototypes and
   restore them into working memory.
3. Structurally consolidate: prune the weakest synapses, restore row norms,
   and — importantly for continual learning — refit the upper read-out on a
   mixture of replayed old and new experience (least squares, no backprop).

The read-out refit is what makes "sleep" reduce catastrophic forgetting in the
online-adaptation experiment: the base weights barely move, but the interface
mapping is re-consolidated on both old and new regimes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import torch

from ..config import ConsolidationConfig
from .system import MTDBSystem


@dataclass
class ReplayItem:
    inputs: torch.Tensor       # [window_len, input_dim]
    surprise: float
    regime: int


class ReplayBuffer:
    """Replay buffer with a per-regime quota.

    Quotas matter: a single FIFO would let the most recent regime wash out the
    older one, and sleep replay would then have nothing old to consolidate.
    """

    def __init__(self, cfg: ConsolidationConfig):
        self.cfg = cfg
        self._by_regime: Dict[int, Deque[ReplayItem]] = {}
        self.total_added = 0

    def _capacity_per_regime(self) -> int:
        n_regimes = max(1, len(self._by_regime))
        return max(1, int(self.cfg.buffer_size // max(2, n_regimes)))

    def add(self, inputs: torch.Tensor, surprise: float, regime: int = 0) -> None:
        if inputs.shape[0] < self.cfg.window_len:
            return
        regime = int(regime)
        cap = self._capacity_per_regime()
        if regime not in self._by_regime:
            self._by_regime[regime] = deque(maxlen=cap)
        # If a new regime appears, trim existing deques to the new quota.
        for key in list(self._by_regime):
            dq = self._by_regime[key]
            if dq.maxlen != cap:
                self._by_regime[key] = deque(dq, maxlen=cap)
        self._by_regime[regime].append(
            ReplayItem(
                inputs=inputs.detach().cpu().clone(),
                surprise=float(surprise),
                regime=regime,
            )
        )
        self.total_added += 1

    def __len__(self) -> int:
        return sum(len(dq) for dq in self._by_regime.values())

    def n_per_regime(self) -> Dict[int, int]:
        return {k: len(v) for k, v in self._by_regime.items()}

    def clear(self) -> None:
        self._by_regime.clear()

    def sample(self, max_items: Optional[int] = None) -> List[ReplayItem]:
        items: List[ReplayItem] = []
        for dq in self._by_regime.values():
            items.extend(dq)
        if max_items is not None and len(items) > max_items:
            items = items[-max_items:]
        return list(items)


class SleepConsolidator:
    def __init__(self, cfg: ConsolidationConfig, system: MTDBSystem, horizon: int = 5):
        self.cfg = cfg
        self.system = system
        self.horizon = int(horizon)

    @torch.no_grad()
    def run(
        self,
        buffer: ReplayBuffer,
        readout=None,
        ridge_lambda: float = 0.05,
    ) -> Dict[str, float]:
        cfg = self.cfg
        stats: Dict[str, float] = {
            "n_windows_replayed": 0.0,
            "n_samples": 0.0,
            "n_prototypes": 0.0,
            "pruned_synapses": 0.0,
        }
        if not cfg.enabled or len(buffer) == 0:
            return stats

        row_norm_before = float(self.system.dynamics.W.norm(dim=1).mean())
        W_before = self.system.dynamics.W.detach().clone()

        # ---- 1. structural consolidation (before replay so that replay sees
        #         the consolidated dynamics, not stale pre-prune weights) ------
        pruned = self.system.dynamics.prune(cfg.prune_fraction)
        if cfg.renorm:
            self.system.dynamics.normalize_rows()
        stats["pruned_synapses"] = float(pruned)
        stats["row_norm_before"] = row_norm_before
        stats["row_norm_after"] = float(self.system.dynamics.W.norm(dim=1).mean())
        stats["weight_drift_l2"] = float((self.system.dynamics.W - W_before).norm())

        # ---- 2. replay: prototypes + read-out training samples --------------
        player = self.system.attractor
        wm = self.system.wm
        prototypes = 0
        Xs: List[torch.Tensor] = []
        Ys: List[torch.Tensor] = []
        for _pass in range(max(1, cfg.replay_passes)):
            for item in buffer.sample():
                out = self.system.replay(item.inputs, allow_memory_write=True)
                stats["n_windows_replayed"] += 1.0
                z = out["z"]
                h = out["h"]
                surprise = out["surprise"]
                L = min(z.shape[0], item.inputs.shape[0])
                if player is not None and L > 0:
                    k = min(3, L)
                    top = torch.topk(surprise[:L], k).indices
                    for idx in top.tolist():
                        if player.write(h[idx]):
                            prototypes += 1
                    if wm is not None:
                        wm.write(h[top[0]])
                if readout is not None and cfg.readout_refit:
                    n = min(L, item.inputs.shape[0] - self.horizon)
                    if n > 0:
                        Xs.append(z[:n])
                        Ys.append(item.inputs[self.horizon : self.horizon + n])
        stats["n_prototypes"] = float(prototypes)

        # ---- 3. replay-gated read-out consolidation ------------------------
        # Only accept the re-fit if it actually lowers replay prediction error;
        # a consolidation pass must not be able to make the interface worse.
        if readout is not None and cfg.readout_refit and Xs:
            X = torch.cat(Xs, dim=0)
            Y = torch.cat(Ys, dim=0)
            stats["n_samples"] = float(X.shape[0])
            if X.shape[0] >= 16:
                if hasattr(readout, "predict_batch"):
                    old_pred = readout.predict_batch(X)
                    old_mse = float(torch.mean((old_pred - Y) ** 2))
                else:
                    old_mse = float("inf")
                old_state = readout.state_dict()
                if hasattr(readout, "refit"):
                    readout.refit(X, Y, ridge_lambda=ridge_lambda)
                else:
                    readout.fit(X, Y)
                new_pred = readout.predict_batch(X)
                new_mse = float(torch.mean((new_pred - Y) ** 2))
                stats["replay_mse_before"] = old_mse
                stats["replay_mse_after"] = new_mse
                if new_mse <= old_mse:
                    stats["readout_refit"] = 1.0
                else:
                    readout.load_state_dict(old_state)
                    stats["readout_refit"] = 0.0
            else:
                stats["readout_refit"] = 0.0

        if self.system.attractor is not None:
            ret = self.system.attractor.retrieval_accuracy(
                cfg.retrieval_samples, cfg.retrieval_noise
            )
            stats.update({f"retrieval_{k}": float(v) for k, v in ret.items()})
        self.system.reset_state()
        return stats
