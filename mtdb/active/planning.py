"""Belief-space planning utilities for active inference.

The class posterior can come from any model (MTDB ensembles or a Transformer
baseline). Planning only needs:

* a class posterior over ``C`` hypotheses,
* a probe→bit table ``code[c, q] ∈ {0,1}``,
* a per-probe noise table.

One-step planning is exact expected-information-gain (expected posterior
entropy). Multi-step planning uses particle rollouts: particles are sampled from
the current posterior, future queries are chosen by the greedy policy, and the
first query with the lowest expected terminal entropy is executed.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def _ensure_noise(noise, n_queries: int) -> np.ndarray:
    arr = np.asarray(noise, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        arr = np.full(n_queries, float(arr[0]))
    if arr.size != n_queries:
        raise ValueError(f"noise must have length 1 or {n_queries}, got {arr.size}")
    return np.clip(arr, 1e-4, 0.499)


def likelihood_vector(code: np.ndarray, noise: np.ndarray, query: int, sign: int) -> np.ndarray:
    bit = code[:, query]
    p = noise[query] + (1.0 - 2.0 * noise[query]) * (bit == int(sign))
    return np.asarray(p, dtype=np.float64)


def posterior_update(
    posterior: np.ndarray,
    code: np.ndarray,
    noise: np.ndarray,
    query: int,
    outcome: float,
) -> np.ndarray:
    sign = 1 if outcome > 0 else 0
    post = np.asarray(posterior, dtype=np.float64) * likelihood_vector(
        code, noise, query, sign
    )
    return post / max(post.sum(), 1e-12)


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    return float(-(p * np.log(p + 1e-12)).sum())


def expected_entropy(
    posterior: np.ndarray, code: np.ndarray, noise: np.ndarray, query: int
) -> float:
    total = 0.0
    for sign in (0, 1):
        like = likelihood_vector(code, noise, query, sign)
        p_outcome = float((posterior * like).sum())
        if p_outcome <= 1e-12:
            continue
        post = posterior * like
        post /= post.sum()
        total += p_outcome * entropy(post)
    return total


def greedy_query(posterior: np.ndarray, code: np.ndarray, noise: np.ndarray) -> int:
    scores = [expected_entropy(posterior, code, noise, q) for q in range(code.shape[1])]
    return int(np.argmin(scores))


def plan_query(
    posterior: np.ndarray,
    code: np.ndarray,
    noise,
    depth: int = 1,
    n_particles: int = 64,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Choose the next probe.

    depth=1  exact one-step expected information gain.
    depth>1  particle rollouts with greedy continuation; the first query is
             scored by expected terminal entropy.
    """
    post = np.asarray(posterior, dtype=np.float64)
    post = post / max(post.sum(), 1e-12)
    code = np.asarray(code, dtype=np.int64)
    noise = _ensure_noise(noise, code.shape[1])
    if depth <= 1:
        return greedy_query(post, code, noise)

    rng = rng or np.random.default_rng(0)
    n_particles = max(8, int(n_particles))
    n_classes = code.shape[0]
    best_query, best_score = 0, float("inf")
    for first in range(code.shape[1]):
        total = 0.0
        for _ in range(n_particles):
            cls = int(rng.choice(n_classes, p=post))
            # first probe outcome for the sampled class
            sign = int(code[cls, first] != 0 and rng.random() > noise[first])
            if code[cls, first] == 0:
                sign = int(rng.random() < noise[first])
            post2 = posterior_update(post, code, noise, first, 1.0 if sign else -1.0)
            for _step in range(1, depth):
                q = greedy_query(post2, code, noise)
                bit = int(code[cls, q])
                correct = rng.random() > noise[q]
                out_bit = bit if correct else 1 - bit
                post2 = posterior_update(post2, code, noise, q, 1.0 if out_bit else -1.0)
            total += entropy(post2)
        score = total / n_particles
        if score < best_score:
            best_score, best_query = score, first
    return int(best_query)


def infer_code_and_noise(
    classes: Sequence[int],
    queries: Sequence[int],
    outcomes: Sequence[float],
    n_classes: int,
    n_queries: int,
    noise_prior: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Learn a probe→bit table and per-probe noise from labelled episodes."""
    counts = np.zeros((n_classes, n_queries, 2), dtype=np.float64)
    for c, q, o in zip(classes, queries, outcomes):
        counts[int(c), int(q), 0 if o < 0 else 1] += 1.0
    code = (counts[:, :, 1] >= counts[:, :, 0]).astype(np.int64)
    flips = np.zeros(n_queries)
    total = np.zeros(n_queries)
    for c, q, o in zip(classes, queries, outcomes):
        bit = int(code[int(c), int(q)])
        total[int(q)] += 1
        flips[int(q)] += int((o > 0) != bool(bit))
    noise = np.where(total > 0, flips / np.maximum(total, 1.0), noise_prior)
    return code, np.clip(noise, 0.01, 0.49)
