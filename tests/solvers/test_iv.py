"""Tests del inversor de volatilidad implícita Black-Scholes.

E1 mide ``MAE_IV``; sin una inversión estable sobre todo el rango
``[σ_min, σ_max]``, esa métrica sería ruido. Este test recorre tres
niveles (bajo, medio y alto) para verificar que la combinación
Newton + bisección converge en colas y centro.
"""

import pytest

from src.solvers import BlackScholesSolver, ImpliedVolatilityInverter


@pytest.mark.parametrize("volatility", [0.05, 0.2, 0.8])
def test_implied_volatility_recovers_multiple_volatility_levels(volatility: float) -> None:
    """Ida y vuelta ``σ → C → σ`` en niveles bajo, medio y alto."""
    solver = BlackScholesSolver()
    inverter = ImpliedVolatilityInverter()

    price = solver.call_price(120.0, 100.0, 1.5, 0.02, volatility)
    implied = inverter.solve_call(price, 120.0, 100.0, 1.5, 0.02)

    assert implied == pytest.approx(volatility, abs=1e-8)
