"""Active-sensing environment used by the v0.2 active-inference benchmark.

The task is deliberately small and exactly specified so that "thinking" can be
measured rather than asserted:

* there are 8 hidden classes, encoded by 3 bits;
* there are 8 binary tests (probes); each test returns one bit of the class
  code corrupted by 10% label noise;
* only 3 well-chosen tests are needed to identify the class, but the test
  matrix contains redundant tests, so *which* tests you query matters;
* the agent sees a query one-hot and the noisy outcome, and has to decide which
  test to run next.

The exact Bayesian active policy is available as an oracle upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SensingEpisode:
    features: np.ndarray          # [T, F] query one-hot + noisy outcome + step
    label: int
    queries: np.ndarray           # [T]
    outcomes: np.ndarray          # [T] in {-1, +1}


class ActiveSensingEnv:
    def __init__(
        self,
        n_classes: int = 8,
        n_queries: int = 8,
        noise: float = 0.1,
        train_horizon: int = 8,
        code: Optional[np.ndarray] = None,
    ):
        if n_classes != 8 or n_queries != 8:
            raise ValueError("this benchmark is defined for 8 classes x 8 tests")
        self.n_classes = int(n_classes)
        self.n_queries = int(n_queries)
        self.noise = float(noise)
        self.query_noise = np.full(self.n_queries, self.noise, dtype=np.float64)
        self.drift_step: Optional[int] = None
        self.drift_queries = set()
        self.drift_noise = 0.4
        self.train_horizon = int(train_horizon)
        self.feature_dim = 2 * self.n_queries + 2
        self.code = self._default_code() if code is None else np.asarray(code, dtype=np.int64)

    # ------------------------------------------------------------------ code
    @staticmethod
    def _default_code() -> np.ndarray:
        bits = np.array([[(c >> i) & 1 for i in range(3)] for c in range(8)], dtype=np.int64)
        b0, b1, b2 = bits[:, 0], bits[:, 1], bits[:, 2]
        # q0/q1 are redundant; q2=b1, q3=b2; remaining columns are parity checks.
        return np.stack(
            [b0, b0, b1, b2, b0 ^ b1, b0 ^ b2, b1 ^ b2, b0 ^ b1 ^ b2], axis=1
        )

    # --------------------------------------------------------------- features
    def feature(self, query: int, outcome: float, step: int, horizon: int) -> np.ndarray:
        f = np.zeros(self.feature_dim, dtype=np.float32)
        f[query] = outcome
        f[self.n_queries + query] = 1.0
        f[2 * self.n_queries] = 1.0
        f[2 * self.n_queries + 1] = step / max(1, horizon - 1)
        return f

    def configure_drift(
        self,
        switch_step: int,
        drift_queries: Sequence[int],
        drift_noise: float = 0.4,
    ) -> None:
        """Make selected probes unreliable after ``switch_step`` ticks."""
        self.drift_step = int(switch_step)
        self.drift_queries = set(int(q) for q in drift_queries)
        self.drift_noise = float(drift_noise)

    def clear_drift(self) -> None:
        self.drift_step = None
        self.drift_queries = set()

    def noise_for(self, query: int, step: Optional[int] = None) -> float:
        if (
            self.drift_step is not None
            and step is not None
            and step >= self.drift_step
            and int(query) in self.drift_queries
        ):
            return self.drift_noise
        return float(self.query_noise[int(query)])

    def observe(
        self,
        label: int,
        query: int,
        rng: np.random.Generator,
        step: Optional[int] = None,
    ) -> float:
        bit = int(self.code[label, query])
        if rng.random() < self.noise_for(query, step):
            bit = 1 - bit
        return 1.0 if bit else -1.0

    # ------------------------------------------------------------- episodes
    def random_episode(self, rng: np.random.Generator, horizon: Optional[int] = None) -> SensingEpisode:
        T = int(horizon or self.train_horizon)
        label = int(rng.integers(0, self.n_classes))
        queries = rng.integers(0, self.n_queries, size=T).astype(np.int64)
        outcomes = np.asarray(
            [self.observe(label, int(q), rng, step=t) for t, q in enumerate(queries)],
            dtype=np.float32,
        )
        feats = np.stack(
            [self.feature(int(q), float(o), t, T) for t, (q, o) in enumerate(zip(queries, outcomes))]
        )
        return SensingEpisode(features=feats, label=label, queries=queries, outcomes=outcomes)

    def fixed_episode(
        self, rng: np.random.Generator, horizon: int, offset: int = 0
    ) -> SensingEpisode:
        T = int(horizon)
        label = int(rng.integers(0, self.n_classes))
        queries = np.asarray([(offset + t) % self.n_queries for t in range(T)], dtype=np.int64)
        outcomes = np.asarray(
            [self.observe(label, int(q), rng, step=t) for t, q in enumerate(queries)],
            dtype=np.float32,
        )
        feats = np.stack(
            [self.feature(int(q), float(o), t, T) for t, (q, o) in enumerate(zip(queries, outcomes))]
        )
        return SensingEpisode(features=feats, label=label, queries=queries, outcomes=outcomes)

    # --------------------------------------------------------- exact Bayes
    def posterior(self, history: Sequence[Tuple[int, float]]) -> np.ndarray:
        post = np.ones(self.n_classes, dtype=np.float64) / self.n_classes
        for q, o in history:
            bit = self.code[:, q]
            sign = 1 if o > 0 else 0
            p = self.noise + (1.0 - 2.0 * self.noise) * (bit == sign)
            post = post * p
            post /= max(post.sum(), 1e-12)
        return post

    def expected_entropy(self, post: np.ndarray, query: int) -> float:
        total = 0.0
        for sign in (1, 0):
            bit = self.code[:, query]
            p = self.noise + (1.0 - 2.0 * self.noise) * (bit == sign)
            po = float((post * p).sum())
            post_o = post * p
            post_o /= max(po, 1e-12)
            total += po * float(-(post_o * np.log(post_o + 1e-12)).sum())
        return total

    def bayes_choice(self, history: Sequence[Tuple[int, float]]) -> int:
        post = self.posterior(history)
        return int(np.argmin([self.expected_entropy(post, q) for q in range(self.n_queries)]))

    def bayes_episode(self, rng: np.random.Generator, horizon: int) -> SensingEpisode:
        T = int(horizon)
        label = int(rng.integers(0, self.n_classes))
        history: List[Tuple[int, float]] = []
        queries, outcomes = [], []
        for t in range(T):
            q = self.bayes_choice(history)
            o = self.observe(label, q, rng, step=t)
            history.append((q, o))
            queries.append(q)
            outcomes.append(o)
        feats = np.stack(
            [self.feature(int(q), float(o), t, T) for t, (q, o) in enumerate(zip(queries, outcomes))]
        )
        return SensingEpisode(
            features=feats,
            label=label,
            queries=np.asarray(queries, dtype=np.int64),
            outcomes=np.asarray(outcomes, dtype=np.float32),
        )

    def bayes_predict(self, episode: SensingEpisode) -> int:
        history = [(int(q), float(o)) for q, o in zip(episode.queries, episode.outcomes)]
        return int(np.argmax(self.posterior(history)))

    # ------------------------------------------------------------- batches
    def random_batch(self, n: int, rng: np.random.Generator, horizon: Optional[int] = None) -> List[SensingEpisode]:
        return [self.random_episode(rng, horizon) for _ in range(n)]

    @staticmethod
    def stack_features(episodes: Sequence[SensingEpisode], horizon: Optional[int] = None) -> np.ndarray:
        T = horizon or min(e.features.shape[0] for e in episodes)
        return np.stack([e.features[:T] for e in episodes])
