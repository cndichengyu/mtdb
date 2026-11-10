import torch

from mtdb.training.readout import RLSReadout, RidgeReadout
from mtdb.utils import set_seed


def test_ridge_fits_linear_map():
    set_seed(0)
    X = torch.randn(200, 12)
    W = torch.randn(12, 3)
    Y = X @ W + 0.3
    ro = RidgeReadout(12, 3, ridge_lambda=1e-6).fit(X, Y)
    pred = ro.predict(X)
    assert torch.allclose(pred, Y, atol=1e-3)


def test_rls_converges_and_matches_ridge():
    set_seed(1)
    X = torch.randn(300, 8)
    W = torch.randn(8, 2)
    Y = X @ W + 0.1
    rls = RLSReadout(8, 2, forget=0.999)
    for _ in range(2):
        for x, y in zip(X, Y):
            rls.update(x, y)
    err = (rls.predict_batch(X) - Y).pow(2).mean()
    assert float(err) < 0.02
    ridge = RidgeReadout(8, 2).fit(X, Y)
    rls2 = RLSReadout(8, 2)
    rls2.set_weights(ridge.w, ridge.b, ridge.x_mean, ridge.x_std)
    assert torch.allclose(rls2.predict_batch(X), ridge.predict(X), atol=1e-4)
    rls2.refit(X, Y, ridge_lambda=1e-6)
    assert float((rls2.predict_batch(X) - Y).pow(2).mean()) < 1e-4
