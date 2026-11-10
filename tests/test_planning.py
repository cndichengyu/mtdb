"""Tests for belief-space planning and calibration utilities."""

import numpy as np

from mtdb.active import planning
from mtdb.active.sensing import ActiveSensingEnv
from mtdb.active.agent import ActiveSensingAgent


def test_planning_depth1_and_depth2():
    env = ActiveSensingEnv()
    post = np.ones(8) / 8
    noise = np.full(8, 0.1)
    q1 = planning.plan_query(post, env.code, noise, depth=1)
    q2 = planning.plan_query(
        post, env.code, noise, depth=2, n_particles=16,
        rng=np.random.default_rng(0),
    )
    assert 0 <= q1 < 8 and 0 <= q2 < 8
    assert planning.entropy(post) > planning.entropy(
        planning.posterior_update(post, env.code, noise, q1, 1.0)
    )


def test_infer_probe_table():
    env = ActiveSensingEnv()
    rng = np.random.default_rng(0)
    episodes = env.random_batch(100, rng, horizon=8)
    labels = [e.label for e in episodes for _ in range(8)]
    queries = [int(q) for e in episodes for q in e.queries]
    outcomes = [float(o) for e in episodes for o in e.outcomes]
    code, noise = planning.infer_code_and_noise(labels, queries, outcomes, 8, 8)
    assert code.shape == (8, 8)
    assert np.all((code == 0) | (code == 1))
    assert noise.shape == (8,) and np.all(noise > 0)


def test_agent_calibration_and_drift(tiny_cfg):
    tiny_cfg.include_sensory_trace = True
    env = ActiveSensingEnv()
    rng = np.random.default_rng(3)
    train = env.random_batch(12, rng, horizon=4)
    agent = ActiveSensingAgent(tiny_cfg, env, seed=0, n_ensemble=2)
    agent.fit(train, passes=1)
    agent.calibrate(train)
    agent.conformal_calibrate(train, alpha=0.1)
    report = agent.calibration_report(train)
    assert "ece" in report and "nll" in report
    assert 0.0 <= report.get("coverage", 0.0) <= 1.0
    env.configure_drift(2, (0, 1), drift_noise=0.4)
    assert env.noise_for(0, 3) == 0.4
    assert env.noise_for(7, 3) == 0.1
    env.clear_drift()
