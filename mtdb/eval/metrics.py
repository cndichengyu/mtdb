"""Evaluation metrics (no scikit-learn dependency)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean((a.float() - b.float()) ** 2))


def nmse(a: torch.Tensor, b: torch.Tensor) -> float:
    """MSE normalised by the variance of the target."""
    a = a.float()
    b = b.float()
    var = float(b.var())
    return float(torch.mean((a - b) ** 2)) / max(var, 1e-8)


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor, dim: int = -1) -> float:
    pred = logits.argmax(dim=dim)
    gold = targets.argmax(dim=dim)
    return float((pred == gold).float().mean())


def masked_accuracy(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, dim: int = -1
) -> float:
    pred = pred[mask]
    target = target[mask]
    if pred.numel() == 0:
        return 0.0
    return float((pred.argmax(dim=dim) == target.argmax(dim=dim)).float().mean())


def auc_roc(scores: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor) -> float:
    """Rank-based ROC-AUC with tie handling (Mann-Whitney U)."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg])
    order = np.argsort(all_scores, kind="mergesort")
    sorted_scores = all_scores[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[i : j + 1] = avg_rank
        i = j + 1
    full_ranks = np.empty_like(ranks)
    full_ranks[order] = ranks
    rank_pos = full_ranks[: pos.size].sum()
    u = rank_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def roc_curve(
    scores: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor, n_points: int = 64
):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(bool)
    thresholds = np.quantile(scores, np.linspace(0, 1, n_points))
    thresholds = np.unique(thresholds)[::-1]
    tpr, fpr = [], []
    n_pos = max(1, labels.sum())
    n_neg = max(1, (~labels).sum())
    for thr in thresholds:
        pred = scores >= thr
        tpr.append(float((pred & labels).sum()) / n_pos)
        fpr.append(float((pred & ~labels).sum()) / n_neg)
    return np.array(fpr), np.array(tpr)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).mean())


def summarize_mse_per_sequence(
    system, readout, seqs: List[dict], horizon: int, allow_memory_write: bool = False
) -> Dict[str, float]:
    """Roll the system over sequences and score the upper read-out."""
    errs = []
    for seq in seqs:
        x = seq["x"]
        system.reset_state()
        zs = []
        for t in range(x.shape[0]):
            out = system.step(x[t], learn=False, allow_memory_write=allow_memory_write)
            zs.append(out.packet.z)
        Z = torch.stack(zs)
        Yhat = readout.predict(Z)
        Y = seq["y"]
        errs.append(mse(Yhat, Y))
    return {
        "mse": float(np.mean(errs)),
        "mse_std": float(np.std(errs)),
        "n_sequences": len(errs),
    }
