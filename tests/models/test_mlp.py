import pytest
import torch

from src.models import MLP


def test_mlp_returns_expected_output_shape_for_each_activation() -> None:
    features = torch.rand(5, 4)

    for activation in ("relu", "softplus", "swish", "tanh"):
        model = MLP(input_dim=4, hidden_width=8, hidden_layers=2, activation=activation)
        assert model(features).shape == (5, 1)


def test_mlp_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="activation"):
        MLP(input_dim=4, activation="gelu")
