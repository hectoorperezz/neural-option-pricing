import numpy as np
import pytest

from src.solvers import BlackScholesSolver, HestonSolver


def test_call_price_and_delta_match_black_scholes_limit() -> None:
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


def test_call_delta_matches_central_finite_differences() -> None:
    solver = HestonSolver(absolute_tolerance=1e-9, relative_tolerance=1e-9)
    grid = [
        {
            "spot": 80.0,
            "strike": 100.0,
            "maturity": 0.25,
            "rate": 0.01,
            "v0": 0.09,
            "theta": 0.04,
            "kappa": 2.0,
            "xi": 0.6,
            "rho": -0.5,
        },
        {
            "spot": 100.0,
            "strike": 100.0,
            "maturity": 1.0,
            "rate": 0.03,
            "v0": 0.04,
            "theta": 0.04,
            "kappa": 1.5,
            "xi": 0.4,
            "rho": -0.7,
        },
        {
            "spot": 130.0,
            "strike": 100.0,
            "maturity": 2.0,
            "rate": 0.05,
            "v0": 0.0225,
            "theta": 0.09,
            "kappa": 3.0,
            "xi": 0.8,
            "rho": -0.2,
        },
    ]

    errors = []
    for params in grid:
        bump = max(1e-3 * params["spot"], 1e-2)
        up = params | {"spot": params["spot"] + bump}
        down = params | {"spot": params["spot"] - bump}
        finite_difference_delta = (solver.call_price(**up) - solver.call_price(**down)) / (
            2.0 * bump
        )
        errors.append(abs(solver.call_delta(**params) - finite_difference_delta))

    assert np.mean(errors) < 1e-4
    assert max(errors) < 1e-4


def test_solver_broadcasts_vector_inputs() -> None:
    solver = HestonSolver()

    spots = np.array([90.0, 100.0, 110.0])
    prices = solver.call_price(spots, 100.0, 1.0, 0.03, 0.04, 0.04, 1.5, 0.4, -0.7)
    deltas = solver.call_delta(spots, 100.0, 1.0, 0.03, 0.04, 0.04, 1.5, 0.4, -0.7)

    assert prices.shape == (3,)
    assert deltas.shape == (3,)
    assert np.all(np.diff(prices) > 0.0)
    assert np.all((0.0 < deltas) & (deltas < 1.0))


def test_solver_rejects_invalid_inputs() -> None:
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

    with pytest.raises(ValueError, match="spot"):
        solver.call_price(**(params | {"spot": 0.0}))
    with pytest.raises(ValueError, match="strike"):
        solver.call_price(**(params | {"strike": 0.0}))
    with pytest.raises(ValueError, match="maturity"):
        solver.call_price(**(params | {"maturity": 0.0}))
    with pytest.raises(ValueError, match="v0"):
        solver.call_price(**(params | {"v0": -0.01}))
    with pytest.raises(ValueError, match="theta"):
        solver.call_price(**(params | {"theta": 0.0}))
    with pytest.raises(ValueError, match="kappa"):
        solver.call_price(**(params | {"kappa": 0.0}))
    with pytest.raises(ValueError, match="xi"):
        solver.call_price(**(params | {"xi": 0.0}))
    with pytest.raises(ValueError, match="rho"):
        solver.call_price(**(params | {"rho": -1.0}))
