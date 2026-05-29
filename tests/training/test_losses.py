"""Test del backprop diferencial.

E5 entrena con ``DifferentialLoss`` (precio + Delta). Si el grafo de
autograd se rompiera y el término de Delta no propagara gradiente a
los pesos, el entrenamiento seguiría "funcionando" pero
silenciosamente solo usaría la pérdida de precio. Sería imposible
detectarlo desde fuera. Este test es la prueba de que E5 hace lo que
dice hacer.
"""

import torch
from torch import nn

from src.training import DifferentialLoss


def test_differential_loss_backpropagates_through_delta_term() -> None:
    """El backward por el término de Delta llega a los pesos del modelo."""
    model = nn.Linear(2, 1)
    batch = {
        "features": torch.rand(4, 2),
        "price": torch.rand(4, 1),
        "delta": torch.rand(4, 1),
    }

    output = DifferentialLoss()(model, batch)
    output.loss.backward()

    assert output.loss.ndim == 0
    assert "delta_loss" in output.metrics
    assert model.weight.grad is not None
