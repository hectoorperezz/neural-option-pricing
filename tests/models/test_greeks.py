"""Test del fundamento de las Greeks autograd.

La Delta del surrogate se calcula derivando respecto a ``m_norm`` y
dividiendo por la anchura del rango de moneyness. Si esa corrección
de la regla de la cadena estuviera mal, todos los ``MAE_Delta`` del
proyecto saldrían escalados por un factor constante sin que ningún
otro test ni los E2E lo detectaran.
"""

import torch
from torch import nn

from src.models import surrogate_delta


def test_surrogate_delta_applies_moneyness_chain_rule() -> None:
    """Modelo lineal con peso ``width`` en la columna de moneyness ⇒ Delta = 1."""
    moneyness_range = (0.4, 2.0)
    width = moneyness_range[1] - moneyness_range[0]
    model = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
        model.weight[0, 0] = width

    features = torch.rand(6, 3)
    delta = surrogate_delta(model, features, moneyness_range=moneyness_range)

    assert torch.allclose(delta, torch.ones(6, 1))
