"""Active-inference agent: mental rollouts + expected free energy.

The agent owns an MTDB base and two bootstrap-ridge readouts trained without
backpropagation:

* a class readout  P(class | z)  (ensemble -> uncertainty);
* an outcome readout P(outcome | z, query) for a candidate probe.

At every decision step it performs a **mental rollout** for each candidate
query: both possible outcomes are hallucinated at the MTDB level, the resulting
latent states are scored by expected class entropy (expected free energy), and
the query with the lowest score is executed. This is the concrete difference
from next-token prediction: the model simulates consequences of its own
questions before acting.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..config import ExperimentConfig
from ..core.system import MTDBSystem
from ..training.readout import RidgeReadout
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
    def proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([m.predict(x.unsqueeze(0))[0] for m in self.members])
        return torch.softmax(logits.mean(dim=0), dim=-1)

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
        code_counts = torch.zeros(
            self.env.n_classes, self.env.n_queries, 2, dtype=torch.float64
        )
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
                    code_counts[ep.label, q, 0 if o < 0 else 1] += 1.0
                    z = self.observe(q, o, t, len(ep.features))
                    class_X.append(z)
                    class_Y.append(
                        torch.nn.functional.one_hot(
                            torch.tensor(ep.label), self.env.n_classes
                        ).float()
                    )
                    z_prev = z
            self.class_ensemble.fit(torch.stack(class_X), torch.stack(class_Y))
            self.outcome_ensemble.fit(torch.stack(out_X), torch.stack(out_Y))
        # learn the probe->bit table and noise rate from labelled training data
        self.code_hat = (code_counts[:, :, 1] >= code_counts[:, :, 0]).long()
        disagreements = []
        for ep in episodes:
            for q, o in zip(ep.queries, ep.outcomes):
                bit = int(self.code_hat[ep.label, int(q)])
                disagreements.append(int((o > 0) != bool(bit)))
        self.noise_hat = float(np.mean(disagreements)) if disagreements else 0.1
        return self

    # --------------------------------------------------------------- predict
    @torch.no_grad()
    def class_probs(self, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self._z if z is None else z
        if not self.class_ensemble.members:
            return torch.ones(self.env.n_classes) / self.env.n_classes
        return self.class_ensemble.proba(z)

    @torch.no_grad()
    def outcome_probs(self, z: torch.Tensor, query: int) -> Tuple[float, float]:
        qoh = torch.zeros(self.env.n_queries)
        qoh[query] = 1.0
        if not self.outcome_ensemble.members:
            return 0.5, 0.5
        p = self.outcome_ensemble.proba(torch.cat([z, qoh]))
        return float(p[0]), float(p[1])

    @torch.no_grad()
    def predict_episode(self, episode: SensingEpisode) -> int:
        self.reset()
        z = None
        for t, feat in enumerate(episode.features):
            out = self.system.step(
                torch.tensor(feat), learn=False, allow_memory_write=False
            )
            z = out.packet.z
        return int(self.class_probs(z).argmax().item())

    def predict_proba_episode(self, episode: SensingEpisode) -> np.ndarray:
        self.reset()
        z = None
        for feat in episode.features:
            out = self.system.step(
                torch.tensor(feat), learn=False, allow_memory_write=False
            )
            z = out.packet.z
        return self.class_probs(z).numpy()

    # ---------------------------------------------------------------- planning
    @torch.no_grad()
    def choose_query(self, step: int, horizon: int) -> int:
        """Expected-free-energy lookahead over candidate probes.

        The posterior over classes comes from the MTDB interface (learned by
        local ridge readouts). The agent simulates the two possible outcomes of
        each candidate probe in *belief space* using the probe table learned
        from labelled training experience, and picks the probe with the lowest
        expected posterior entropy. If no probe table is available it falls
        back to latent rollouts through the MTDB dynamics.
        """
        class_probs = self.class_probs().detach().cpu().double()
        best_query, best_score = 0, float("inf")
        if self.code_hat is not None:
            for query in range(self.env.n_queries):
                expected_entropy = 0.0
                for sign in (0, 1):
                    bit = self.code_hat[:, query]
                    likelihood = self.noise_hat + (1.0 - 2.0 * self.noise_hat) * (
                        bit == sign
                    ).double()
                    p_outcome = float((class_probs * likelihood).sum())
                    if p_outcome <= 1e-9:
                        continue
                    post = class_probs * likelihood
                    post = post / post.sum().clamp_min(1e-12)
                    expected_entropy += p_outcome * float(
                        -(post * torch.log(post + 1e-12)).sum()
                    )
                if expected_entropy < best_score:
                    best_score, best_query = expected_entropy, query
            return best_query

        # fallback: latent rollout with the learned outcome readout
        z = self._z
        for query in range(self.env.n_queries):
            p_neg, p_pos = self.outcome_probs(z, query)
            expected_entropy = 0.0
            for outcome, prob in ((-1.0, p_neg), (1.0, p_pos)):
                if prob <= 1e-6:
                    continue
                snap = self.system.snapshot_state(light=True)
                feat = torch.tensor(self.env.feature(query, outcome, step, horizon))
                out = self.system.step(
                    feat, learn=False, allow_memory_write=False
                )
                probs = self.class_probs(out.packet.z)
                entropy = float(-(probs * torch.log(probs + 1e-9)).sum())
                self.system.restore_state(snap)
                expected_entropy += prob * entropy
            if expected_entropy < best_score:
                best_score, best_query = expected_entropy, query
        return best_query

    def plan_episode(self, rng: np.random.Generator, horizon: int) -> SensingEpisode:
        """Run the active policy in the environment and return the episode."""
        self.reset()
        label = int(rng.integers(0, self.env.n_classes))
        queries: List[int] = []
        outcomes: List[float] = []
        feats: List[np.ndarray] = []
        for t in range(horizon):
            query = self.choose_query(t, horizon)
            outcome = self.env.observe(label, query, rng)
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
