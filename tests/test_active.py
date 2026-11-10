"""Tests for the v0.2 active-inference layer."""

import numpy as np
import torch

from mtdb.active.agent import ActiveSensingAgent
from mtdb.active.sensing import ActiveSensingEnv
from mtdb.baselines.transformer import (
    CausalTransformerClassifier,
    train_transformer_classifier,
    transformer_predict,
)


def test_sensing_env_contract():
    env = ActiveSensingEnv()
    assert env.code.shape == (8, 8)
    assert env.feature_dim == 18
    rng = np.random.default_rng(0)
    ep = env.random_episode(rng, horizon=5)
    assert ep.features.shape == (5, 18)
    assert ep.queries.shape == (5,)
    assert set(np.unique(ep.outcomes)).issubset({-1.0, 1.0})
    # exact Bayes can at least identify more than chance with 5 active probes
    episodes = [env.bayes_episode(rng, 5) for _ in range(20)]
    hits = [env.bayes_predict(ep) == ep.label for ep in episodes]
    assert np.mean(hits) > 0.4


def test_agent_fit_and_plan(tiny_cfg):
    tiny_cfg.include_sensory_trace = True
    env = ActiveSensingEnv()
    rng = np.random.default_rng(1)
    train = env.random_batch(8, rng, horizon=4)
    agent = ActiveSensingAgent(tiny_cfg, env, seed=0, n_ensemble=2)
    agent.fit(train, passes=1)
    assert agent.code_hat is not None
    assert agent._z.shape[0] > 18
    ep = agent.plan_episode(rng, horizon=2)
    assert ep.features.shape == (2, 18)
    assert all(0 <= int(q) < 8 for q in ep.queries)
    pred = agent.predict_episode(ep)
    assert 0 <= pred < 8


def test_transformer_smoke(tiny_cfg):
    env = ActiveSensingEnv()
    rng = np.random.default_rng(2)
    train = env.random_batch(6, rng, horizon=3)
    model = CausalTransformerClassifier(
        feature_dim=18, n_classes=8, d_model=16, nhead=2,
        num_layers=1, dim_feedforward=32,
    )
    logits = model(torch.stack([torch.tensor(e.features) for e in train]))
    assert logits.shape == (6, 3, 8)
    # a couple of gradient steps should run
    model = train_transformer_classifier(
        train, feature_dim=18, n_classes=8, epochs=2, batch_size=4
    )
    preds = transformer_predict(model, train)
    assert len(preds) == len(train)
