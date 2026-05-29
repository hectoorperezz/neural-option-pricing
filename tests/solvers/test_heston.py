"""Tests del solver Heston por Fourier semi-cerrado.

Dos propiedades teóricas que QuantLib no cubre por sí solo:

* el límite de Black-Scholes cuando ``ξ → 0`` con ``v0 = θ = σ²``;
* la reconstrucción ``C = S·P_1 - K·e^{-rT}·P_2`` con ``Delta = P_1``,
  que es el fundamento del experimento E5.
"""

import pytest

from src.solvers import BlackScholesSolver, HestonSolver


def test_call_price_and_delta_match_black_scholes_limit() -> None:
    """Con ``ξ`` casi nulo y varianza determinista, Heston ≈ Black-Scholes."""
    bs_solver = BlackScholesSolver()
    heston_solver = HestonSolver(absolute_tolerance=1e-9, relative_tolerance=1e-9)

    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.03
    volatility = 0.2
    variance = volatility * volatility

    heston_price = heston_solver.call_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        v0=variance,
        theta=variance,
        kappa=8.0,
        xi=1e-3,
        rho=-0.3,
    )
    heston_delta = heston_solver.call_delta(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        v0=variance,
        theta=variance,
        kappa=8.0,
        xi=1e-3,
        rho=-0.3,
    )

    bs_price = bs_solver.call_price(spot, strike, maturity, rate, volatility)
    bs_delta = bs_solver.call_delta(spot, strike, maturity, rate, volatility)

    assert heston_price == pytest.approx(bs_price, abs=1e-4)
    assert heston_delta == pytest.approx(bs_delta, abs=1e-4)


def test_call_probabilities_reconstruct_call_price_and_delta() -> None:
    """``P_1`` y ``P_2`` reconstruyen el precio y, con ``q=0``, dan Delta = ``P_1``."""
    import numpy as np

    solver = HestonSolver()
    params = {
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 1.0,
        "rate": 0.03,
        "v0": 0.04,
        "theta": 0.04,
        "kappa": 1.5,
        "xi": 0.4,
        "rho": -0.7,
    }

    p1, p2 = solver.call_probabilities(**params)
    price = solver.call_price(**params)
    delta = solver.call_delta(**params)
    reconstructed_price = (
        params["spot"] * p1
        - params["strike"] * np.exp(-params["rate"] * params["maturity"]) * p2
    )

    assert 0.0 < p2 < p1 < 1.0
    assert price == pytest.approx(reconstructed_price, abs=1e-10)
    assert delta == pytest.approx(p1, abs=1e-12)
