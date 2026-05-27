from __future__ import annotations

import torch
from torch import nn


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
    "swish": Swish,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    """Configurable feed-forward surrogate."""

    def __init__(
        self,
        input_dim: int,
        hidden_width: int = 128,
        hidden_layers: int = 4,
        activation: str = "swish",
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be strictly positive")
        if hidden_width <= 0:
            raise ValueError("hidden_width must be strictly positive")
        if hidden_layers <= 0:
            raise ValueError("hidden_layers must be strictly positive")
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation: {activation}")

        activation_cls = ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_width))
            layers.append(activation_cls())
            current_dim = hidden_width
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
