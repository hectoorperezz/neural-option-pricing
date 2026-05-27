import pytest

from src.solvers import BlackScholesSolver, ImpliedVolatilityInverter


def test_implied_volatility_recovers_reference_volatility() -> None:
    solver = BlackScholesSolver()
    inverter = ImpliedVolatilityInverter()

    price = solver.call_price(100.0, 105.0, 0.75, 0.03, 0.35)
    implied = inverter.solve_call(price, 100.0, 105.0, 0.75, 0.03)

    assert implied == pytest.approx(0.35, abs=1e-9)


@pytest.mark.parametrize("volatility", [0.05, 0.2, 0.8])
def test_implied_volatility_recovers_multiple_volatility_levels(volatility: float) -> None:
    solver = BlackScholesSolver()
    inverter = ImpliedVolatilityInverter()

    price = solver.call_price(120.0, 100.0, 1.5, 0.02, volatility)
    implied = inverter.solve_call(price, 120.0, 100.0, 1.5, 0.02)

    assert implied == pytest.approx(volatility, abs=1e-8)


def test_implied_volatility_rejects_prices_outside_no_arbitrage_bounds() -> None:
    inverter = ImpliedVolatilityInverter()

    with pytest.raises(ValueError, match="bounds"):
        inverter.solve_call(101.0, 100.0, 100.0, 1.0, 0.01)


@pytest.mark.parametrize(
    "args",
    [
        (float("nan"), 100.0, 100.0, 1.0, 0.01),
        (10.0, float("nan"), 100.0, 1.0, 0.01),
        (10.0, 100.0, float("nan"), 1.0, 0.01),
        (10.0, 100.0, 100.0, float("nan"), 0.01),
        (10.0, 100.0, 100.0, 1.0, float("nan")),
    ],
)
def test_implied_volatility_rejects_non_finite_inputs(args: tuple[float, ...]) -> None:
    inverter = ImpliedVolatilityInverter()

    with pytest.raises(ValueError, match="finite"):
        inverter.solve_call(*args)
