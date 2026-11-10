import torch

from mtdb.tasks.synthetic import (
    FEATURE_DIM,
    DelayedRecallWorld,
    MultimodalWorld,
    make_prediction_set,
)


def test_multimodal_shapes_and_values():
    world = MultimodalWorld(horizon=3, seed=0)
    seq = world.sample(seq_len=64, regime="A")
    assert seq["x"].shape == (64, FEATURE_DIM)
    assert seq["y"].shape == (64, FEATURE_DIM)
    assert torch.isfinite(seq["x"]).all()
    assert seq["x"][:, 6:10].sum(dim=1).min() > 0.99
    # target is input shifted by horizon
    assert torch.allclose(seq["y"][:-3], seq["x"][3:])


def test_anomalies_are_injected():
    world = MultimodalWorld(horizon=2, anomaly_rate=0.05, seed=1)
    seqs = make_prediction_set(world, 4, seq_len=96, anomalies=True)
    assert all(seq["anomaly"].any() for seq in seqs)


def test_delayed_recall_world():
    world = DelayedRecallWorld(n_classes=8, cue_len=3, delay=20, tail=10, seed=2)
    seq = world.sample()
    T = seq["x"].shape[0]
    assert T == 33
    assert seq["y"].shape == (T, FEATURE_DIM)
    # cue visible only at the beginning
    assert seq["x"][:3].argmax(dim=1).eq(seq["cue"]).all()
    assert seq["x"][3:, :8].abs().sum() == 0
    assert seq["eval_mask"].sum() == 10
