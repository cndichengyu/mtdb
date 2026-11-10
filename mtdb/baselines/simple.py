"""Simple non-neural / non-Transformer baselines for the prediction task.

* Persistence: predict that nothing changes.
* Linear AR(k): ridge regression on the last k raw observation frames.
* Mean predictor: always predict the training mean.

These are deliberately cheap: if the MTDB base does not beat them, something is
wrong with the base. (A GRU baseline can be added in ``baselines/gru.py`` for
downstream comparisons without changing the base.)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from ..eval.metrics import mse


def persistence(seq: dict, horizon: int) -> torch.Tensor:
    return seq["x"].clone()


def mean_predictor(train_seqs: List[dict]) -> torch.Tensor:
    xs = torch.cat([s["x"] for s in train_seqs], dim=0)
    return xs.mean(dim=0)


class LinearAR:
    def __init__(self, order: int = 4, ridge_lambda: float = 0.05):
        self.order = int(order)
        self.ridge_lambda = float(ridge_lambda)
        self.w = None
        self.b = None
        self.feat_mean = None
        self.feat_std = None

    def _features(self, seq: dict) -> torch.Tensor:
        x = seq["x"]
        T, F = x.shape
        k = self.order
        pad = torch.zeros(k - 1, F)
        xp = torch.cat([pad, x], dim=0)  # [T+k-1, F]
        feats = []
        for i in range(k):
            feats.append(xp[i : i + T])
        return torch.cat(feats, dim=1)  # [T, k*F]

    def fit(self, seqs: List[dict]) -> "LinearAR":
        X = torch.cat([self._features(s) for s in seqs], dim=0)
        Y = torch.cat([s["y"] for s in seqs], dim=0)
        self.feat_mean = X.mean(dim=0)
        self.feat_std = X.std(dim=0).clamp_min(1e-6)
        Xs = (X - self.feat_mean) / self.feat_std
        n, d = Xs.shape
        A = Xs.T @ Xs + self.ridge_lambda * n * torch.eye(d)
        B = Xs.T @ (Y - Y.mean(dim=0))
        self.w = torch.linalg.solve(A, B)
        self.b = Y.mean(dim=0)
        return self

    def predict(self, seq: dict) -> torch.Tensor:
        X = (self._features(seq) - self.feat_mean) / self.feat_std
        return X @ self.w + self.b

    def evaluate(self, seqs: List[dict]) -> Dict[str, float]:
        errs = [mse(self.predict(s), s["y"]) for s in seqs]
        return {"mse": float(np.mean(errs)), "mse_std": float(np.std(errs))}


def evaluate_direct(preds: List[torch.Tensor], seqs: List[dict]) -> Dict[str, float]:
    errs = [mse(p, s["y"]) for p, s in zip(preds, seqs)]
    return {"mse": float(np.mean(errs)), "mse_std": float(np.std(errs))}
