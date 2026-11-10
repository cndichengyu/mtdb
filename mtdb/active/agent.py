"""Active-inference agent with calibrated uncertainty and optional multi-step planning.

The agent owns an MTDB base and bootstrap-ridge readouts trained without
backpropagation:

* a class readout  P(class | z)  (ensemble; temperature-calibrated);
* an outcome readout P(outcome | z, query);
* a probe table learned from labelled training experience.

At each decision step it plans in belief space. Depth-1 planning is exact
expected information gain; depth>1 uses particle rollouts and a greedy
continuation policy. The agent also supports online per-probe reliability
updates (for non-stationary noise) and conformal prediction sets.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..config import ExperimentConfig
from ..core.system import MTDBSystem
from ..training.readout import RidgeReadout
from . import planning
from .sensing import ActiveSensingEnv, SensingEpisode


class BootstrapRidgeEnsemble:
    """A tiny ensemble of ridge readouts, trained by least squares only."""

    def __init__(
        self,
        n_members: int = 5,
        ridge_lambda: float = 0.1,
        seed: int = 0,
    ):
        self.n_members = int(n_members)
        self.ridge_lambda = float(ridge_lambda)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.members: List[RidgeReadout] = []

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> "BootstrapRidgeEnsemble":
        n = X.shape[0]
        self.members = []
        for _ in range(self.n_members):
            idx = torch.randint(0, n, (n,), generator=self.generator)
            self.members.append(
                RidgeReadout(X.shape[1], Y.shape[1], self.ridge_lambda).fit(
                    X[idx], Y[idx]
                )
            )
        return self

    @torch.no_grad()
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        if not self.members:
            return torch.zeros(x.shape[0] if x.dim() > 1 else 1, 0) if x.dim() > 1 else torch.zeros(0)
        return torch.stack([m.predict(x.unsqueeze(0))[0] for m in self.members]).mean(dim=0)

    @torch.no_grad()
    def proba(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        logits = self.logits(x)
        if logits.numel() == 0:
            return torch.ones(8) / 8
        return torch.softmax(logits / max(float(temperature), 1e-3), dim=-1)

    def state_dict(self) -> List[Dict[str, torch.Tensor]]:
        return [m.state_dict() for m in self.members]

    def load_state_dict(self, state: List[Dict[str, torch.Tensor]]) -> None:
        self.members = []
        for item in state:
            m = RidgeReadout(len(item["w"]), len(item["b"]))
            m.load_state_dict(item)
            self.members.append(m)


class ActiveSensingAgent:
    def __init__(
        self,
        cfg: ExperimentConfig,
        env: ActiveSensingEnv,
        seed: int = 0,
        n_ensemble: int = 5,
        ridge_lambda: float = 0.1,
    ):
        self.cfg = cfg
        self.env = env
        self.seed = int(seed)
        self.ridge_lambda = float(ridge_lambda)
        self.system = MTDBSystem(
            cfg,
            input_dim=env.feature_dim,
            upper_dim=env.n_classes,
            seed=seed,
            device=cfg.device,
        )
        self.system.plasticity_enabled = False
        self.class_ensemble = BootstrapRidgeEnsemble(
            n_ensemble, ridge_lambda, seed=seed + 1
        )
        self.outcome_ensemble = BootstrapRidgeEnsemble(
            n_ensemble, ridge_lambda, seed=seed + 2
        )
        self._z = torch.zeros(self.system.spec.total_dim)
        self.queries: List[int] = []
        self.outcomes: List[float] = []
        self.code_hat: Optional[torch.Tensor] = None
        self.noise_hat: float = 0.1
        self.noise_per_query: Optional[np.ndarray] = None
        self.temperature: float = 1.0
        self.noise_update_rate: float = 0.2
        self.noise_cap: float = 0.49
        self.conformal_q: Optional[float] = None
        self.planning_rng = np.random.default_rng(seed + 99)
        self._trace_slice = self.system.spec.slices().get("trace")

    # ------------------------------------------------------------------ reset
    def reset(self) -> None:
        self.system.reset_state()
        self._z = torch.zeros(self.system.spec.total_dim)
        self.queries = []
        self.outcomes = []

    def observe(self, query: int, outcome: float, step: int, horizon: int) -> torch.Tensor:
        feat = torch.tensor(self.env.feature(query, outcome, step, horizon))
        out = self.system.step(feat, learn=False, allow_memory_write=False)
        self._z = out.packet.z
        self.queries.append(int(query))
        self.outcomes.append(float(outcome))
        return self._z

    # ------------------------------------------------------------------- fit
    @torch.no_grad()
    def fit(self, episodes: Sequence[SensingEpisode], passes: int = 3) -> "ActiveSensingAgent":
        """Fit the class/outcome ensembles on passively collected episodes."""
        for _pass in range(passes):
            class_X: List[torch.Tensor] = []
            class_Y: List[torch.Tensor] = []
            out_X: List[torch.Tensor] = []
            out_Y: List[torch.Tensor] = []
            for ep in episodes:
                self.reset()
                z_prev = torch.zeros(self.system.spec.total_dim)
                for t, feat in enumerate(ep.features):
                    q = int(ep.queries[t])
                    o = float(ep.outcomes[t])
                    qoh = torch.zeros(self.env.n_queries)
                    qoh[q] = 1.0
                    out_X.append(torch.cat([z_prev, qoh]))
                    out_Y.append(torch.tensor([(1.0 - o) / 2.0, (1.0 + o) / 2.0]))
                    z = self.observe(q, o, t, len(ep.features))
                    class_X.append(self._class_features(z))
                    class_Y.append(
                        torch.nn.functional.one_hot(
                            torch.tensor(ep.label), self.env.n_classes
                        ).float()
                    )
                    z_prev = z
            self.class_ensemble.fit(torch.stack(class_X), torch.stack(class_Y))
            self.outcome_ensemble.fit(torch.stack(out_X), torch.stack(out_Y))

        labels = [ep.label for ep in episodes for _ in range(len(ep.queries))]
        queries = [int(q) for ep in episodes for q in ep.queries]
        outcomes = [float(o) for ep in episodes for o in ep.outcomes]
        code, noise = planning.infer_code_and_noise(
            labels, queries, outcomes, self.env.n_classes, self.env.n_queries
        )
        self.code_hat = torch.as_tensor(code, dtype=torch.long)
        # planning starts from a common reliability prior; per-probe deviations
        # are learned online only when the environment drifts. This avoids
        # overfitting tiny per-probe noise estimates at the very first steps.
        self.noise_hat = float(np.mean(noise))
        self.noise_per_query = np.full(
            self.env.n_queries, self.noise_hat, dtype=np.float64
        )
        self.noise_per_query_est = noise.copy()
        return self

    # --------------------------------------------------------------- predict
    def _class_features(self, z: torch.Tensor) -> torch.Tensor:
        # the optional sensory evidence trace is the cheapest sufficient
        # statistic for evidence accumulation in this task; using it keeps the
        # readout small and well-conditioned
        if self._trace_slice is not None:
            return z[self._trace_slice]
        return z

    @torch.no_grad()
    def class_logits(self, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self._z if z is None else z
        if not self.class_ensemble.members:
            return torch.zeros(self.env.n_classes)
        return self.class_ensemble.logits(self._class_features(z))

    @torch.no_grad()
    def class_probs(self, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._probs_from_logits(self.class_logits(z))

    @torch.no_grad()
    def _probs_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits / max(self.temperature, 1e-3), dim=-1)

    @torch.no_grad()
    def outcome_probs(self, z: torch.Tensor, query: int) -> Tuple[float, float]:
        qoh = torch.zeros(self.env.n_queries)
        qoh[query] = 1.0
        if not self.outcome_ensemble.members:
            return 0.5, 0.5
        p = self.outcome_ensemble.proba(torch.cat([z, qoh]))
        return float(p[0]), float(p[1])

    @torch.no_grad()
    def _final_logits(self, episode: SensingEpisode) -> torch.Tensor:
        self.reset()
        z = None
        for feat in episode.features:
            out = self.system.step(
                torch.tensor(feat), learn=False, allow_memory_write=False
            )
            z = out.packet.z
        return self.class_logits(z)

    @torch.no_grad()
    def predict_episode(self, episode: SensingEpisode) -> int:
        return int(self._probs_from_logits(self._final_logits(episode)).argmax().item())

    def predict_proba_episode(self, episode: SensingEpisode) -> np.ndarray:
        return self._probs_from_logits(self._final_logits(episode)).numpy()

    # ----------------------------------------------------------- calibration
    @torch.no_grad()
    def calibrate(self, episodes: Sequence[SensingEpisode]) -> float:
        """Temperature scaling on a held-out calibration split."""
        logits, labels = [], []
        for ep in episodes:
            logits.append(self._final_logits(ep))
            labels.append(ep.label)
        if not logits:
            return self.temperature
        logits_t = torch.stack(logits)
        labels_t = torch.tensor(labels, dtype=torch.long)
        best_t, best_nll = 1.0, float("inf")
        for temperature in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0):
            log_probs = torch.log_softmax(logits_t / temperature, dim=-1)
            nll = float(-log_probs[torch.arange(len(labels_t)), labels_t].mean())
            if nll < best_nll:
                best_nll, best_t = nll, float(temperature)
        self.temperature = best_t
        return best_t

    def conformal_calibrate(
        self, episodes: Sequence[SensingEpisode], alpha: float = 0.1
    ) -> float:
        """Split-conformal threshold for class prediction sets."""
        scores = []
        for ep in episodes:
            probs = self._probs_from_logits(self._final_logits(ep))
            scores.append(float(1.0 - probs[ep.label]))
        n = len(scores)
        if n == 0:
            return 0.0
        level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
        self.conformal_q = float(np.quantile(scores, level))
        return self.conformal_q

    def prediction_set(self, episode: SensingEpisode) -> List[int]:
        if self.conformal_q is None:
            return [self.predict_episode(episode)]
        probs = self._probs_from_logits(self._final_logits(episode)).numpy()
        selected = np.flatnonzero(probs >= 1.0 - self.conformal_q).tolist()
        if not selected:
            selected = [int(np.argmax(probs))]
        return selected

    def calibration_report(
        self, episodes: Sequence[SensingEpisode], n_bins: int = 10
    ) -> Dict[str, float]:
        """NLL / ECE / conformal coverage+size without using labels for planning."""
        probs, labels = [], []
        for ep in episodes:
            probs.append(self._probs_from_logits(self._final_logits(ep)).numpy())
            labels.append(ep.label)
        if not probs:
            return {}
        probs = np.stack(probs)
        labels = np.asarray(labels)
        n = len(labels)
        nll = float(-np.mean(np.log(probs[np.arange(n), labels] + 1e-12)))
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        ece = 0.0
        for lo in np.linspace(0.0, 1.0, n_bins + 1)[:-1]:
            hi = lo + 1.0 / n_bins
            mask = (conf > lo) & (conf <= hi)
            if mask.sum():
                acc = float((pred[mask] == labels[mask]).mean())
                ece += float(mask.mean()) * abs(acc - float(conf[mask].mean()))
        report = {
            "nll": nll,
            "ece": float(ece),
            "accuracy": float((pred == labels).mean()),
            "temperature": float(self.temperature),
        }
        if self.conformal_q is not None:
            sizes, covered = [], []
            for ep in episodes:
                p = self._probs_from_logits(self._final_logits(ep)).numpy()
                selected = np.flatnonzero(p >= 1.0 - self.conformal_q).tolist()
                if not selected:
                    selected = [int(np.argmax(p))]
                sizes.append(len(selected))
                covered.append(int(ep.label in selected))
            report["coverage"] = float(np.mean(covered))
            report["avg_set_size"] = float(np.mean(sizes))
        return report

    # ---------------------------------------------------------- online noise
    def reset_noise(self, prior: Optional[np.ndarray] = None) -> None:
        if self.noise_per_query is None:
            return
        if prior is None:
            self.noise_per_query = np.full(
                self.env.n_queries, float(self.noise_hat), dtype=np.float64
            )
        else:
            self.noise_per_query = np.asarray(prior, dtype=np.float64).copy()

    def update_noise(self, query: int, outcome: float, prior: np.ndarray) -> None:
        """Posterior-weighted online update of one probe's reliability."""
        if self.code_hat is None or self.noise_per_query is None:
            return
        bit = self.code_hat[:, query].numpy()
        observed = 1 if outcome > 0 else 0
        mismatch = float((prior * (bit != observed)).sum())
        rate = float(self.noise_update_rate)
        self.noise_per_query[query] = float(
            np.clip(
                (1.0 - rate) * self.noise_per_query[query] + rate * mismatch,
                0.01,
                float(self.noise_cap),
            )
        )

    # ---------------------------------------------------------------- planning
    def choose_query(
        self,
        step: int,
        horizon: int,
        depth: int = 1,
        n_particles: int = 64,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """Plan the next probe in belief space."""
        if self.code_hat is not None and self.noise_per_query is not None:
            return planning.plan_query(
                self.class_probs().detach().numpy(),
                self.code_hat.numpy(),
                self.noise_per_query,
                depth=depth,
                n_particles=n_particles,
                rng=rng or self.planning_rng,
            )

        # fallback: latent rollout with the learned outcome readout
        z = self._z
        best_query, best_score = 0, float("inf")
        for query in range(self.env.n_queries):
            p_neg, p_pos = self.outcome_probs(z, query)
            expected_entropy = 0.0
            for outcome, prob in ((-1.0, p_neg), (1.0, p_pos)):
                if prob <= 1e-6:
                    continue
                snap = self.system.snapshot_state(light=True)
                feat = torch.tensor(self.env.feature(query, outcome, step, horizon))
                out = self.system.step(feat, learn=False, allow_memory_write=False)
                probs = self.class_probs(out.packet.z)
                entropy = float(-(probs * torch.log(probs + 1e-9)).sum())
                self.system.restore_state(snap)
                expected_entropy += prob * entropy
            if expected_entropy < best_score:
                best_score, best_query = expected_entropy, query
        return best_query

    def plan_episode(
        self,
        rng: np.random.Generator,
        horizon: int,
        depth: int = 1,
        n_particles: int = 64,
        adaptive_noise: bool = False,
    ) -> SensingEpisode:
        """Run the active policy in the environment and return the episode."""
        self.reset()
        label = int(rng.integers(0, self.env.n_classes))
        queries: List[int] = []
        outcomes: List[float] = []
        feats: List[np.ndarray] = []
        for t in range(horizon):
            prior = self.class_probs().detach().numpy()
            query = self.choose_query(
                t, horizon, depth=depth, n_particles=n_particles, rng=rng
            )
            outcome = self.env.observe(label, query, rng, step=t)
            if adaptive_noise:
                self.update_noise(query, outcome, prior)
            feat = self.env.feature(query, outcome, t, horizon)
            feats.append(feat)
            self.observe(query, outcome, t, horizon)
            queries.append(query)
            outcomes.append(outcome)
        return SensingEpisode(
            features=np.stack(feats),
            label=label,
            queries=np.asarray(queries, dtype=np.int64),
            outcomes=np.asarray(outcomes, dtype=np.float32),
        )

    def fixed_queries(self, horizon: int) -> List[int]:
        return [t % self.env.n_queries for t in range(horizon)]
