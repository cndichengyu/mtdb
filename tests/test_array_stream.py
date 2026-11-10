import numpy as np
import torch

from mtdb.tasks.array import make_streams_from_arrays, streams_from_npz
from mtdb.training.trainer import MTDBTrainer


def _arrays():
    rng = np.random.default_rng(0)
    train = rng.normal(0.0, 2.0, size=(4, 20, 6)).astype(np.float32)
    eval_ = rng.normal(0.0, 2.0, size=(2, 20, 6)).astype(np.float32)
    train_b = rng.normal(0.3, 2.0, size=(3, 20, 6)).astype(np.float32)
    eval_b = rng.normal(0.3, 2.0, size=(2, 20, 6)).astype(np.float32)
    anom = rng.normal(0.0, 2.0, size=(2, 20, 6)).astype(np.float32)
    mask = np.zeros((2, 20), dtype=bool)
    mask[:, 7:9] = True
    return train, eval_, train_b, eval_b, anom, mask


def test_make_streams_shapes_and_normalisation():
    train, eval_, train_b, eval_b, anom, mask = _arrays()
    streams, meta = make_streams_from_arrays(
        train, eval_, train_b, eval_b, anom, mask, horizon=3
    )
    assert meta["input_dim"] == 6
    assert meta["has_regime_b"] and meta["has_anomaly_eval"]
    assert len(streams["train_A"]) == 4
    assert len(streams["eval_A"]) == 2
    x = streams["train_A"][0]["x"]
    y = streams["train_A"][0]["y"]
    assert x.shape == (20, 6) and y.shape == (20, 6)
    assert torch.allclose(y[:-3], x[3:])
    full = torch.cat([s["x"] for s in streams["train_A"]], dim=0)
    assert abs(float(full.mean())) < 0.2
    assert streams["eval_anom"][0]["anomaly"].sum() == 2


def test_streams_from_npz(tmp_path):
    train, eval_, train_b, eval_b, anom, mask = _arrays()
    path = tmp_path / "stream.npz"
    np.savez(path, train=train, eval=eval_, train_b=train_b, eval_b=eval_b,
             eval_anomaly=anom, eval_anomaly_mask=mask)
    streams, meta = streams_from_npz(str(path), horizon=2)
    assert meta["input_dim"] == 6
    assert "eval_anom" in streams


def test_custom_stream_trainer(tiny_cfg, tmp_path):
    train, eval_, train_b, eval_b, anom, mask = _arrays()
    streams, meta = make_streams_from_arrays(
        train, eval_, train_b, eval_b, anom, mask, horizon=2
    )
    tiny_cfg.train.epochs = 1
    tiny_cfg.train.run_stage2 = False
    tiny_cfg.train.forgetting_stress = False
    tiny_cfg.train.run_recall = False
    tiny_cfg.train.ablations = False
    tiny_cfg.train.save_model = False
    trainer = MTDBTrainer(tiny_cfg, streams=streams, run_dir=tmp_path / "run")
    results = trainer.run()
    assert results["input_dim"] == 6
    assert results["task"] == "custom-stream"
    assert results["stage1"]["prediction"]["mse"] > 0
    assert results["stage1"]["anomaly"]["auc"] > 0.0
    assert results["interface_dim"] > 6
