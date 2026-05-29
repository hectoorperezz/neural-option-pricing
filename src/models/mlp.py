"""MLP feed-forward usado como surrogate de pricing.

Una sola arquitectura parametrizable cubre los once surrogates del
proyecto: solo varían la activación y los hiperparámetros. Por defecto,
4 × 128 con Swish.
"""

from __future__ import annotations

import torch
from torch import nn


class Swish(nn.Module):
    """Activación ``x * sigmoid(x)``, versión suave de ReLU."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


# Activaciones comparadas en el experimento E2.
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
    "swish": Swish,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    """Red feed-forward que aproxima el precio normalizado ``y = C/K``.

    Espera los inputs ya normalizados a ``[0, 1]``. La Delta del surrogate
    se obtiene aparte, en :func:`src.models.greeks.surrogate_delta`.

    Args:
        input_dim: Número de variables de entrada tras la normalización.
        hidden_width: Unidades por capa oculta.
        hidden_layers: Número de capas ocultas.
        activation: Clave de :data:`ACTIVATIONS`.
        output_dim: Dimensión de salida; siempre 1 en el proyecto.
    """

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
