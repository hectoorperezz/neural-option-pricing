import numpy as np
import pytest

from src.solvers import BlackScholesSolver


def test_call_price_delta_and_vega_match_reference_values() -> None:
    solver = BlackScholesSolver()

    price = solver.call_price(100.0, 100.0, 1.0, 0.05, 0.2)
    delta = solver.call_delta(100.0, 100.0, 1.0, 0.05, 0.2)
    vega = solver.call_vega(100.0, 100.0, 1.0, 0.05, 0.2)

    assert price == pytest.approx(10.450583572185565, abs=1e-12)
    assert delta == pytest.approx(0.6368306511756191, abs=1e-12)
    assert vega == pytest.approx(37.52403469169379, abs=1e-12)


def test_solver_broadcasts_vector_inputs() -> None:
    solver = BlackScholesSolver()

    spots = np.array([90.0, 100.0, 110.0])
    prices = solver.call_price(spots, 100.0, 1.0, 0.05, 0.2)
    deltas = solver.call_delta(spots, 100.0, 1.0, 0.05, 0.2)

    assert prices.shape == (3,)
    assert deltas.shape == (3,)
    assert np.all(np.diff(prices) > 0.0)
    assert np.all((0.0 < deltas) & (deltas < 1.0))


def test_solver_rejects_invalid_inputs() -> None:
    solver = BlackScholesSolver()

    with pytest.raises(ValueError, match="spot"):
        solver.call_price(0.0, 100.0, 1.0, 0.05, 0.2)
    with pytest.raises(ValueError, match="strike"):
        solver.call_price(100.0, 0.0, 1.0, 0.05, 0.2)
    with pytest.raises(ValueError, match="maturity"):
        solver.call_price(100.0, 100.0, 0.0, 0.05, 0.2)
    with pytest.raises(ValueError, match="volatility"):
        solver.call_price(100.0, 100.0, 1.0, 0.05, 0.0)
