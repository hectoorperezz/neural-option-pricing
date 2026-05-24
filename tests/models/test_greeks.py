import torch
from torch import nn

from src.models import surrogate_delta, surrogate_price_and_delta


def test_surrogate_delta_applies_moneyness_chain_rule() -> None:
    moneyness_range = (0.4, 2.0)
    width = moneyness_range[1] - moneyness_range[0]
    model = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
        model.weight[0, 0] = width

    features = torch.rand(6, 3)
    delta = surrogate_delta(model, features, moneyness_range=moneyness_range)

    assert torch.allclose(delta, torch.ones(6, 1))


def test_surrogate_price_and_delta_keeps_gradient_for_differential_loss() -> None:
    model = nn.Linear(2, 1)
    features = torch.rand(4, 2)

    prices, deltas = surrogate_price_and_delta(model, features, create_graph=True)
    loss = prices.mean() + deltas.mean()
    loss.backward()

    assert model.weight.grad is not None
