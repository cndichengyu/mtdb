"""Read-outs trained WITHOUT backpropagation.

* ``RidgeReadout`` — batch least squares (closed form). This is how the
  unsupervised predictive-coding stage fits the upper read-out. It is a local
  learning rule at the level of the interface: no gradients flow through the
  dynamical core.
* ``RLSReadout`` — recursive least squares for online adaptation. One rank-one
  update per tick, O(D^2), still no backpropagation.

Both operate on the interface vector z(t) and can be checkpointed.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


class RidgeReadout:
    def __init__(self, in_dim: int, out_dim: int, ridge_lambda: float = 0.05):
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.ridge_lambda = float(ridge_lambda)
        self.w = torch.zeros(self.in_dim, self.out_dim)
        self.b = torch.zeros(self.out_dim)
        self.x_mean = torch.zeros(self.in_dim)
        self.x_std = torch.ones(self.in_dim)
        self.precision = None
        self.fitted = False

    def _standardize(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.x_mean) / self.x_std

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> "RidgeReadout":
        X = X.float().cpu()
        Y = Y.float().cpu()
        assert X.shape[1] == self.in_dim, f"expected {self.in_dim}, got {X.shape[1]}"
        self.x_mean = X.mean(dim=0)
        self.x_std = X.std(dim=0).clamp_min(1e-6)
        Xs = self._standardize(X)
        y_mean = Y.mean(dim=0)
        Yc = Y - y_mean
        n = Xs.shape[0]
        A = Xs.T @ Xs
        lam = max(self.ridge_lambda, 1e-8) * n
        A = A + lam * torch.eye(self.in_dim)
        B = Xs.T @ Yc
        self.w = torch.linalg.solve(A, B)
        self.b = y_mean
        if self.in_dim <= 512:
            self.precision = torch.linalg.inv(A)
        self.fitted = True
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("RidgeReadout.predict called before fit()")
        Xs = self._standardize(X.float().cpu())
        return Xs @ self.w + self.b

    def predict_batch(self, X: torch.Tensor) -> torch.Tensor:
        return self.predict(X)

    def set_weights(self, w: torch.Tensor, b: torch.Tensor, x_mean=None, x_std=None) -> None:
        self.w = w.clone()
        self.b = b.clone()
        if x_mean is not None:
            self.x_mean = x_mean.clone()
        if x_std is not None:
            self.x_std = x_std.clone()
        self.fitted = True

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "w": self.w.cpu(),
            "b": self.b.cpu(),
            "x_mean": self.x_mean.cpu(),
            "x_std": self.x_std.cpu(),
        }

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.w = state["w"].clone()
        self.b = state["b"].clone()
        self.x_mean = state["x_mean"].clone()
        self.x_std = state["x_std"].clone()
        self.fitted = True


class RLSReadout:
    """Recursive least squares with exponential forgetting (float64 internals)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        forget: float = 0.999,
        init_delta: float = 100.0,
        theta_norm_cap: float = 200.0,
    ):
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.forget = float(forget)
        self.init_delta = float(init_delta)
        self.theta_norm_cap = float(theta_norm_cap)
        self.theta = torch.zeros(self.in_dim + 1, self.out_dim, dtype=torch.float64)
        self.P = torch.eye(self.in_dim + 1, dtype=torch.float64) * self.init_delta
        self.n_updates = 0
        self.x_mean = torch.zeros(self.in_dim)
        self.x_std = torch.ones(self.in_dim)
        self.fitted = False

    # ---------------------------------------------------------------- predict
    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        z = (x.float().cpu() - self.x_mean) / self.x_std
        return torch.cat([z.double(), torch.ones(1, dtype=torch.float64)])

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            return torch.zeros(self.out_dim)
        return (self.theta.T @ self._phi(x)).float()

    def predict_batch(self, X: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            return torch.zeros(X.shape[0], self.out_dim)
        Xs = (X.float().cpu() - self.x_mean) / self.x_std
        return (Xs.double() @ self.theta[:-1] + self.theta[-1]).float()

    # ----------------------------------------------------------------- update
    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor, forget: Optional[float] = None) -> float:
        phi = self._phi(x)
        y = y.float().cpu().double()
        Pphi = self.P @ phi
        lam = self.forget if forget is None else forget
        denom = lam + phi @ Pphi
        gain = Pphi / denom.clamp_min(1e-9)
        err = y - self.theta.T @ phi
        theta_new = self.theta + gain.unsqueeze(1) * err.unsqueeze(0)
        P_new = (self.P - gain.unsqueeze(1) * Pphi.unsqueeze(0)) / lam

        # ---- numerical guards: RLS can drift under long, correlated streams.
        if not (torch.isfinite(theta_new).all() and torch.isfinite(P_new).all()):
            return 0.0  # skip this update, keep the last good posterior
        if self.n_updates % 16 == 0:
            P_new = 0.5 * (P_new + P_new.T)
        pdiag = P_new.diagonal()
        if not torch.isfinite(pdiag).all() or pdiag.max() > 1e8:
            P_new = torch.eye(self.in_dim + 1, dtype=torch.float64) * self.init_delta
        t_norm = theta_new.norm()
        if t_norm > self.theta_norm_cap:
            theta_new = theta_new * (self.theta_norm_cap / t_norm)
            P_new = torch.eye(self.in_dim + 1, dtype=torch.float64) * self.init_delta
        self.theta = theta_new
        self.P = P_new
        self.P.diagonal().clamp_min_(1e-12)
        self.n_updates += 1
        self.fitted = True
        return float(err.pow(2).mean().sqrt())

    @torch.no_grad()
    def set_weights(
        self,
        w: torch.Tensor,
        b: torch.Tensor,
        x_mean: Optional[torch.Tensor] = None,
        x_std: Optional[torch.Tensor] = None,
        precision: Optional[torch.Tensor] = None,
    ) -> None:
        self.theta[: self.in_dim] = w.double().cpu()
        self.theta[self.in_dim] = b.double().cpu()
        if x_mean is not None:
            self.x_mean = x_mean.float().cpu().clone()
        if x_std is not None:
            self.x_std = x_std.float().cpu().clone()
        if precision is not None:
            P = torch.zeros(self.in_dim + 1, self.in_dim + 1, dtype=torch.float64)
            P[: self.in_dim, : self.in_dim] = precision.double().cpu()
            # small prior on the bias row/column
            P[self.in_dim, self.in_dim] = 1.0 / max(1e-3, float(self.init_delta))
            self.P = P
        else:
            self.P = torch.eye(self.in_dim + 1, dtype=torch.float64) * self.init_delta
        self.fitted = True

    @torch.no_grad()
    def refit(self, X: torch.Tensor, Y: torch.Tensor, ridge_lambda: float = 0.05) -> "RLSReadout":
        """Least-squares consolidation of the readout from replay samples."""
        ridge = RidgeReadout(self.in_dim, self.out_dim, ridge_lambda=ridge_lambda)
        ridge.fit(X, Y)
        self.x_mean = ridge.x_mean.clone()
        self.x_std = ridge.x_std.clone()
        self.theta = torch.cat([ridge.w, ridge.b.unsqueeze(0)]).double()
        self.P = torch.eye(self.in_dim + 1, dtype=torch.float64) * self.init_delta
        self.fitted = True
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "theta": self.theta.cpu(),
            "P": self.P.cpu(),
            "x_mean": self.x_mean.cpu(),
            "x_std": self.x_std.cpu(),
            "n_updates": torch.tensor(self.n_updates),
        }

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.theta = state["theta"].clone()
        self.P = state["P"].clone()
        self.x_mean = state["x_mean"].clone()
        self.x_std = state["x_std"].clone()
        self.n_updates = int(state["n_updates"].item())
        self.fitted = True
