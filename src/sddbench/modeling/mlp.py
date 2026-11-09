"""Minimal configurable MLP for frozen feature vectors."""
from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], n_classes: int,
                 dropout: float = 0.0, activation: str = "relu"):
        super().__init__()
        if activation != "relu":
            raise ValueError(f"unsupported activation: {activation!r}")
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            layers += [nn.Linear(prev, int(width)), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(width)
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
