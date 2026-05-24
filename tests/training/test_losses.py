import torch
from torch import nn

from src.training import DifferentialLoss, PriceLoss


def test_price_loss_returns_scalar_and_metric() -> None:
    model = nn.Linear(2, 1)
    batch = {"features": torch.rand(4, 2), "price": torch.rand(4, 1)}

    output = PriceLoss()(model, batch)
    output.loss.backward()

    assert output.loss.ndim == 0
    assert "price_loss" in output.metrics
    assert model.weight.grad is not None


def test_differential_loss_backpropagates_through_delta_term() -> None:
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
