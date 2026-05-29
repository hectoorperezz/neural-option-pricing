"""Pérdidas usadas en el entrenamiento de los surrogates.

Dos variantes: ``PriceLoss`` para los surrogates clásicos (E1..E4) y
``DifferentialLoss`` para el ``differential machine learning`` de E5,
que añade la Delta autograd al objetivo. Ambas devuelven un
:class:`LossOutput` con la pérdida diferenciable y métricas escalares
ya desconectadas del grafo, listas para logging.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.models.greeks import surrogate_price_and_delta


@dataclass(frozen=True)
class LossOutput:
    """Pérdida total y dict de métricas escalares para logging."""

    loss: torch.Tensor
    metrics: dict[str, float]


class PriceLoss(nn.Module):
    """Pérdida L1 sobre el precio normalizado ``C/K``."""

    def forward(self, model: nn.Module, batch: dict[str, torch.Tensor]) -> LossOutput:
        predictions = model(batch["features"])
        price_loss = F.l1_loss(predictions, batch["price"])
        return LossOutput(loss=price_loss, metrics={"price_loss": price_loss.detach().item()})


class DifferentialLoss(nn.Module):
    """Pérdida conjunta de precio y Delta para E5.

    La Delta se obtiene por autograd con ``create_graph=True`` para que
    sea diferenciable respecto a los parámetros del modelo. El objetivo
    es ``price_weight * L1(precio) + delta_weight * L1(Delta)``.

    Args:
        price_weight: Coeficiente sobre el término de precio.
        delta_weight: Coeficiente sobre el término de Delta.
        moneyness_range: Rango original de ``m`` usado por
            :func:`surrogate_price_and_delta` para deshacer la
            normalización min-max al calcular la Delta.
    """

    def __init__(
        self,
        price_weight: float = 1.0,
        delta_weight: float = 1.0,
        moneyness_range: tuple[float, float] = (0.4, 2.0),
    ) -> None:
        super().__init__()
        if price_weight <= 0.0:
            raise ValueError("price_weight must be strictly positive")
        if delta_weight <= 0.0:
            raise ValueError("delta_weight must be strictly positive")
        self.price_weight = price_weight
        self.delta_weight = delta_weight
        self.moneyness_range = moneyness_range

    def forward(self, model: nn.Module, batch: dict[str, torch.Tensor]) -> LossOutput:
        if "delta" not in batch:
            raise ValueError("DifferentialLoss requires delta labels")

        predictions, predicted_delta = surrogate_price_and_delta(
            model,
            batch["features"],
            moneyness_range=self.moneyness_range,
            create_graph=True,
        )
        price_loss = F.l1_loss(predictions, batch["price"])
        delta_loss = F.l1_loss(predicted_delta, batch["delta"])
        loss = self.price_weight * price_loss + self.delta_weight * delta_loss
        return LossOutput(
            loss=loss,
            metrics={
                "price_loss": price_loss.detach().item(),
                "delta_loss": delta_loss.detach().item(),
            },
        )
