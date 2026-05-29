"""Bucle de entrenamiento estándar de los surrogates.

Bucle minimalista que comparten todos los experimentos: Adam,
``L1`` sobre precio (y opcionalmente Delta), validación cada época y
selección de mejor checkpoint por ``MAE(C/K)``. Los entrenamientos
diferenciales (E5) usan el mismo bucle porque el create_graph vive en
:class:`~src.training.losses.DifferentialLoss`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.losses import LossOutput


@dataclass(frozen=True)
class TrainConfig:
    """Hiperparámetros generales del entrenamiento.

    Sirve como contrato común entre los scripts ``train_*.py`` y el
    :class:`Trainer`. Los valores por defecto son los de E1.
    """

    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "cpu"


@dataclass
class Trainer:
    """Trainer mínimo con selección de mejor checkpoint por validación.

    Después de cada época evalúa ``MAE(C/K)`` en validación y guarda el
    ``state_dict`` cuando mejora; ``load_best`` restaura ese estado al
    terminar.
    """

    model: nn.Module
    loss_fn: nn.Module
    optimizer: torch.optim.Optimizer
    train_loader: DataLoader
    validation_loader: DataLoader
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.model.to(self.device)
        self.best_state_dict: dict[str, torch.Tensor] | None = None
        self.best_validation_mae = float("inf")

    def fit(
        self,
        epochs: int,
        on_epoch_end: Callable[[dict[str, float]], None] | None = None,
    ) -> list[dict[str, float]]:
        """Entrena ``epochs`` épocas y devuelve el histórico de métricas.

        Args:
            epochs: Número de épocas a ejecutar.
            on_epoch_end: Callback opcional invocado al final de cada
                época con el dict de métricas (útil para barras de
                progreso o logging externo).
        """
        if epochs <= 0:
            raise ValueError("epochs must be strictly positive")

        history: list[dict[str, float]] = []
        for epoch in range(epochs):
            train_metrics = self._train_epoch()
            validation_mae = self.evaluate_price_mae()
            if validation_mae < self.best_validation_mae:
                self.best_validation_mae = validation_mae
                self.best_state_dict = deepcopy(self.model.state_dict())
            history.append(
                record := {
                    "epoch": float(epoch + 1),
                    "validation_price_mae": validation_mae,
                    **train_metrics,
                }
            )
            if on_epoch_end is not None:
                on_epoch_end(record)
        return history

    def load_best(self) -> None:
        """Restaura el modelo al checkpoint con menor ``MAE(C/K)`` de validación."""
        if self.best_state_dict is None:
            raise RuntimeError("no best checkpoint is available")
        self.model.load_state_dict(self.best_state_dict)

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        total_items = 0

        for batch in self.train_loader:
            batch = self._move_batch(batch)
            self.optimizer.zero_grad()
            output: LossOutput = self.loss_fn(self.model, batch)
            output.loss.backward()
            self.optimizer.step()

            batch_size = int(batch["features"].shape[0])
            total_items += batch_size
            totals["loss"] = totals.get("loss", 0.0) + output.loss.detach().item() * batch_size
            for name, value in output.metrics.items():
                totals[name] = totals.get(name, 0.0) + value * batch_size

        return {name: value / total_items for name, value in totals.items()}

    @torch.no_grad()
    def evaluate_price_mae(self) -> float:
        """``MAE`` sobre el precio normalizado ``C/K`` en el set de validación."""
        self.model.eval()
        total_error = 0.0
        total_items = 0
        for batch in self.validation_loader:
            batch = self._move_batch(batch)
            predictions = self.model(batch["features"])
            errors = torch.abs(predictions - batch["price"])
            total_error += errors.sum().item()
            total_items += int(errors.numel())
        return total_error / total_items

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            name: value.to(self.device)
            for name, value in batch.items()
            if name != "raw_inputs"
        }
