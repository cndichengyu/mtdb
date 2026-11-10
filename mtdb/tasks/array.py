"""Adapters for training MTDB on your own feature streams.

The base only needs two things from a modality front-end:

* a feature matrix ``x[t, :]`` for every tick (any sensor embedding / audio
  feature / event histogram works), and
* a prediction target, which this module derives as ``x[t + horizon]``.

The helper below converts raw arrays (or an ``.npz`` file) into the same
sequence-dict format used by the synthetic experiments, so the rest of the
training pipeline works unchanged.

``.npz`` layout
---------------
Required keys (arrays of shape ``[N, T, F]`` or a single ``[T, F]``):

* ``train``  — training stream, used for stage 1 + readout fitting
* ``eval``   — held-out stream, same feature dimension ``F``

Optional keys:

* ``train_b`` / ``eval_b`` — a second regime for the stage-2 online experiment.
  If omitted, stage 2 runs on ``train_b = train`` (documented in the logs).
* ``eval_anomaly`` — held-out stream with injected anomalies (only for the
  anomaly-detection probe).
* ``eval_anomaly_mask`` — boolean array ``[N, T]`` marking anomaly ticks.

Features are z-scored using *training-split statistics only* (no leakage) and
clipped to ±8; targets are computed after normalisation.

Note: sleep replay stores whole sequences, so each sequence should be at least
``consolidation.window_len`` ticks long (96 ticks in the debug preset, 128 in
the standard preset). Shorter sequences still train, but sleep replay will have
no windows to store. Lower ``consolidation.window_len`` for short streams.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

__all__ = ["make_streams_from_arrays", "streams_from_npz"]


def _as_3d(array, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape [N, T, F] or [T, F], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


def _as_mask_3d(mask, n: int, t: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim == 1:
        arr = np.broadcast_to(arr[None, :], (n, t))
    if arr.shape != (n, t):
        raise ValueError(f"anomaly mask must have shape [{n}, {t}], got {arr.shape}")
    return arr


def _shifted_targets(seq: np.ndarray, horizon: int) -> np.ndarray:
    T = seq.shape[0]
    idx = np.clip(np.arange(T) + horizon, 0, T - 1)
    return seq[idx]


def make_streams_from_arrays(
    train,
    eval,
    train_b=None,
    eval_b=None,
    eval_anomaly=None,
    eval_anomaly_mask=None,
    horizon: int = 5,
    normalize: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Dict[str, List[dict]], Dict[str, object]]:
    """Build trainer-ready sequence lists from raw feature arrays.

    Returns ``(streams, meta)``. ``streams`` contains ``train_A`` / ``eval_A``
    and, when provided, ``train_B`` / ``eval_B`` / ``eval_anom``. Each sequence
    dict has ``x``, ``y`` (both ``[T, F]``), and optionally ``anomaly``.
    """
    horizon = int(horizon)
    train_arr = _as_3d(train, "train")
    eval_arr = _as_3d(eval, "eval")
    F = train_arr.shape[-1]
    if eval_arr.shape[-1] != F:
        raise ValueError(
            f"train/eval feature dims differ: {F} vs {eval_arr.shape[-1]}"
        )

    arrays: Dict[str, np.ndarray] = {"train": train_arr, "eval": eval_arr}
    if train_b is not None:
        arrays["train_b"] = _as_3d(train_b, "train_b")
    if eval_b is not None:
        arrays["eval_b"] = _as_3d(eval_b, "eval_b")
    if eval_anomaly is not None:
        arrays["eval_anomaly"] = _as_3d(eval_anomaly, "eval_anomaly")
    for key, arr in arrays.items():
        if arr.shape[-1] != F:
            raise ValueError(f"{key} feature dim {arr.shape[-1]} != train dim {F}")

    mean = train_arr.reshape(-1, F).mean(axis=0)
    std = train_arr.reshape(-1, F).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    if normalize:
        for key in arrays:
            arrays[key] = np.clip((arrays[key] - mean) / std, -8.0, 8.0)

    def build(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> List[dict]:
        seqs: List[dict] = []
        for i in range(arr.shape[0]):
            x = torch.as_tensor(arr[i], dtype=dtype)
            y = torch.as_tensor(_shifted_targets(arr[i], horizon), dtype=dtype)
            seq = {"x": x, "y": y}
            if mask is not None:
                seq["anomaly"] = torch.as_tensor(mask[i], dtype=torch.bool)
            seqs.append(seq)
        return seqs

    streams: Dict[str, List[dict]] = {
        "train_A": build(arrays["train"]),
        "eval_A": build(arrays["eval"]),
    }
    streams["train_B"] = build(arrays.get("train_b", arrays["train"]))
    streams["eval_B"] = build(arrays.get("eval_b", arrays["eval"]))
    if "eval_anomaly" in arrays and eval_anomaly_mask is not None:
        n, T, _ = arrays["eval_anomaly"].shape
        mask = _as_mask_3d(eval_anomaly_mask, n, T)
        streams["eval_anom"] = build(arrays["eval_anomaly"], mask)

    meta = {
        "input_dim": int(F),
        "horizon": horizon,
        "normalize": bool(normalize),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "n_train": int(arrays["train"].shape[0]),
        "n_eval": int(arrays["eval"].shape[0]),
        "has_regime_b": train_b is not None or eval_b is not None,
        "has_anomaly_eval": "eval_anom" in streams,
    }
    return streams, meta


def streams_from_npz(
    path: str,
    horizon: int = 5,
    normalize: bool = True,
) -> Tuple[Dict[str, List[dict]], Dict[str, object]]:
    """Load an ``.npz`` file and build trainer-ready streams."""
    data = np.load(path, allow_pickle=False)
    required = {"train", "eval"}
    missing = required - set(data.files)
    if missing:
        raise KeyError(
            f"{path} is missing required keys {sorted(missing)}; "
            "expected arrays 'train' and 'eval'"
        )
    return make_streams_from_arrays(
        train=data["train"],
        eval=data["eval"],
        train_b=data["train_b"] if "train_b" in data.files else None,
        eval_b=data["eval_b"] if "eval_b" in data.files else None,
        eval_anomaly=data["eval_anomaly"] if "eval_anomaly" in data.files else None,
        eval_anomaly_mask=(
            data["eval_anomaly_mask"] if "eval_anomaly_mask" in data.files else None
        ),
        horizon=horizon,
        normalize=normalize,
    )
