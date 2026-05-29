"""Delta del surrogate por autograd, con corrección de min-max.

La red trabaja con la moneyness normalizada ``m_norm``; aplicamos la
regla de la cadena para volver a la escala financiera ``m = S/K``.
"""

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
    """Calcula precio y Delta en una sola pasada con autograd.

    La Delta se obtiene derivando la salida respecto a la moneyness
    normalizada y dividiendo por la anchura del rango, equivalente a
    ``∂ŷ/∂m`` en la escala original.

    Args:
        model: Surrogate entrenado, en modo evaluación o entrenamiento.
        features: Tensor ``(N, d)`` con los inputs ya normalizados.
        moneyness_range: Mínimo y máximo de ``m`` usados en la
            normalización del dataset.
        moneyness_index: Columna de ``features`` que corresponde a la
            moneyness; por convención del proyecto es la 0.
        create_graph: ``True`` cuando la Delta se va a usar dentro de la
            pérdida (E5, differential ML); ``False`` para evaluación.
    """

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
    """Devuelve solo la Delta; descarta el precio."""
    _, deltas = surrogate_price_and_delta(
        model,
        features,
        moneyness_range=moneyness_range,
        moneyness_index=moneyness_index,
        create_graph=create_graph,
    )
    return deltas
