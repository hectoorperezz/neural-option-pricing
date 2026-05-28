from __future__ import annotations

import torch
from torch import nn


def surrogate_price_and_delta(
    model: nn.Module,
    features: torch.Tensor,
    moneyness_range: tuple[float, float] = (0.4, 2.0),
    moneyness_index: int = 0,
    create_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Devuelve precio y Delta corrigiendo la normalización min-max de moneyness."""

    if moneyness_range[1] <= moneyness_range[0]:
        raise ValueError("moneyness_range must satisfy min < max")

    differentiable_features = features.detach().clone().requires_grad_(True)
    prices = model(differentiable_features)
    gradients = torch.autograd.grad(
        outputs=prices,
        inputs=differentiable_features,
        grad_outputs=torch.ones_like(prices),
        create_graph=create_graph,
        retain_graph=create_graph,
        only_inputs=True,
    )[0]
    moneyness_width = moneyness_range[1] - moneyness_range[0]
    deltas = gradients[:, moneyness_index : moneyness_index + 1] / moneyness_width
    return prices, deltas


def surrogate_delta(
    model: nn.Module,
    features: torch.Tensor,
    moneyness_range: tuple[float, float] = (0.4, 2.0),
    moneyness_index: int = 0,
    create_graph: bool = False,
) -> torch.Tensor:
    _, deltas = surrogate_price_and_delta(
        model,
        features,
        moneyness_range=moneyness_range,
        moneyness_index=moneyness_index,
        create_graph=create_graph,
    )
    return deltas
