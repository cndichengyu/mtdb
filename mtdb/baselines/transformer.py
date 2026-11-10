"""A small causal Transformer baseline for the active-sensing benchmark.

This is a *baseline only* — the point of the repo is to study a non-Transformer
substrate. The Transformer is trained with normal backpropagation on the same
passively collected episodes and is evaluated on exactly the same observation
sequences as MTDB, including actively selected ones.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import numpy as np
from torch import nn

from ..active.sensing import SensingEpisode


class CausalTransformerClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        n_classes: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        max_len: int = 16,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.max_len = max_len
        self.inp = nn.Linear(feature_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self.register_buffer("_causal_mask", self._mask(max_len), persistent=False)

    @staticmethod
    def _mask(T: int) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        T = features.shape[1]
        mask = self._causal_mask[:T, :T]
        h = self.inp(features) + self.pos[:, :T]
        h = self.encoder(h, mask=mask)
        return self.head(self.norm(h))


def stack_episodes(episodes: Sequence[SensingEpisode]) -> torch.Tensor:
    return torch.as_tensor(
        np.stack([e.features for e in episodes]), dtype=torch.float32
    )


def train_transformer_classifier(
    episodes: Sequence[SensingEpisode],
    feature_dim: int,
    n_classes: int,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
) -> CausalTransformerClassifier:
    torch.manual_seed(seed)
    model = CausalTransformerClassifier(feature_dim, n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    X = stack_episodes(episodes)
    y = torch.tensor([e.label for e in episodes], dtype=torch.long)
    n = X.shape[0]
    for _epoch in range(epochs):
        perm = torch.randperm(n)
        model.train()
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits = model(X[idx])
            # class target is shared across all positions; train on every token
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, n_classes),
                y[idx].repeat_interleave(logits.shape[1]),
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def transformer_predict(
    model: CausalTransformerClassifier, episodes: Sequence[SensingEpisode]
) -> List[int]:
    if not episodes:
        return []
    T = min(e.features.shape[0] for e in episodes)
    X = torch.as_tensor(
        np.stack([e.features[:T] for e in episodes]), dtype=torch.float32
    )
    logits = model(X)[:, -1]
    return logits.argmax(dim=-1).tolist()


def transformer_param_count(model: CausalTransformerClassifier) -> int:
    return sum(p.numel() for p in model.parameters())
