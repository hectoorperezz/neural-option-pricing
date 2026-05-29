"""Test de la corrección min-max en la ruta de evaluación por lotes.

``tests/models/test_greeks.py`` verifica la regla de la cadena en el
helper directo ``surrogate_delta``. En evaluación real se usa
``predict_surrogate_prices_and_deltas``, que itera por lotes y mueve
tensores entre devices. Este test garantiza que la corrección
``∂ŷ/∂m_norm × 1/width`` también se aplica por esa ruta; sin ella, la
Delta reportada por los reports estaría escalada de forma incorrecta.
"""

import numpy as np
import torch
from torch import nn

from src.evaluation.metrics import predict_surrogate_prices_and_deltas


class _TinyMLP(nn.Module):
    def __init__(self, input_dim: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_predict_delta_matches_manual_chain_rule_on_linear_model() -> None:
    """Modelo lineal ``y = w0·m_norm + ...`` ⇒ ``Delta = w0 / (m_max - m_min)``."""
    torch.manual_seed(0)
    model = _TinyMLP(input_dim=4)
    weights = model.linear.weight.detach().numpy().reshape(-1)
    w0 = float(weights[0])
    moneyness_range = (0.4, 2.0)
    expected_delta = w0 / (moneyness_range[1] - moneyness_range[0])

    features = torch.zeros(5, 4)
    _, deltas = predict_surrogate_prices_and_deltas(
        model, features, batch_size=5, device="cpu", moneyness_range=moneyness_range
    )

    np.testing.assert_allclose(deltas, np.full(5, expected_delta), atol=1e-6)
