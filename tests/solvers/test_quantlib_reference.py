import QuantLib as ql
import pytest

from src.solvers import BlackScholesSolver, HestonSolver


REFERENCE_DATE = ql.Date(1, 1, 2026)
DAY_COUNT = ql.Actual365Fixed()
HESTON_INTEGRATION_ORDER = 192


def quantlib_black_scholes_call(
    spot: float,
    strike: float,
    maturity_days: int,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> tuple[float, float, float]:
    ql.Settings.instance().evaluationDate = REFERENCE_DATE
    maturity_date = REFERENCE_DATE + maturity_days
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call, strike),
        ql.EuropeanExercise(maturity_date),
    )
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        ql.YieldTermStructureHandle(
            ql.FlatForward(
                REFERENCE_DATE, dividend_yield, DAY_COUNT, ql.Continuous, ql.NoFrequency
            )
        ),
        ql.YieldTermStructureHandle(
            ql.FlatForward(REFERENCE_DATE, rate, DAY_COUNT, ql.Continuous, ql.NoFrequency)
        ),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(REFERENCE_DATE, ql.NullCalendar(), volatility, DAY_COUNT)
        ),
    )
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option.NPV(), option.delta(), option.vega()


def quantlib_heston_call_price(
    spot: float,
    strike: float,
    maturity_days: int,
    rate: float,
    v0: float,
    theta: float,
    kappa: float,
    xi: float,
    rho: float,
    dividend_yield: float = 0.0,
) -> float:
    ql.Settings.instance().evaluationDate = REFERENCE_DATE
    maturity_date = REFERENCE_DATE + maturity_days
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call, strike),
        ql.EuropeanExercise(maturity_date),
    )
    risk_free_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(REFERENCE_DATE, rate, DAY_COUNT, ql.Continuous, ql.NoFrequency)
    )
    dividend_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(REFERENCE_DATE, dividend_yield, DAY_COUNT, ql.Continuous, ql.NoFrequency)
    )
    process = ql.HestonProcess(
        risk_free_curve,
        dividend_curve,
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        v0,
        kappa,
        theta,
        xi,
        rho,
    )
    model = ql.HestonModel(process)
    option.setPricingEngine(ql.AnalyticHestonEngine(model, HESTON_INTEGRATION_ORDER))
    return option.NPV()


BLACK_SCHOLES_REFERENCE_CASES = [
    pytest.param(100.0, 100.0, 365, 0.03, 0.20, 0.00, id="atm_1y"),
    pytest.param(80.0, 100.0, 90, 0.01, 0.35, 0.00, id="otm_3m_high_vol"),
    pytest.param(130.0, 100.0, 730, 0.05, 0.25, 0.02, id="itm_2y_with_dividend"),
    pytest.param(60.0, 100.0, 30, 0.02, 0.80, 0.00, id="deep_otm_1m_stress_vol"),
    pytest.param(180.0, 100.0, 7, 0.075, 0.30, 0.00, id="deep_itm_weekly_high_rate"),
    pytest.param(100.0, 100.0, 7, 0.00, 0.03, 0.00, id="atm_weekly_low_vol"),
    pytest.param(100.0, 100.0, 730, 0.00, 1.00, 0.00, id="atm_2y_max_vol"),
]


@pytest.mark.parametrize(
    (
        "spot",
        "strike",
        "maturity_days",
        "rate",
        "volatility",
        "dividend_yield",
    ),
    BLACK_SCHOLES_REFERENCE_CASES,
)
def test_black_scholes_matches_quantlib_price_delta_and_vega(
    spot: float,
    strike: float,
    maturity_days: int,
    rate: float,
    volatility: float,
    dividend_yield: float,
) -> None:
    solver = BlackScholesSolver()
    maturity = maturity_days / 365.0

    quantlib_price, quantlib_delta, quantlib_vega = quantlib_black_scholes_call(
        spot, strike, maturity_days, rate, volatility, dividend_yield
    )
    price = solver.call_price(spot, strike, maturity, rate, volatility, dividend_yield)
    delta = solver.call_delta(spot, strike, maturity, rate, volatility, dividend_yield)
    vega = solver.call_vega(spot, strike, maturity, rate, volatility, dividend_yield)

    assert price == pytest.approx(quantlib_price, abs=1e-10)
    assert delta == pytest.approx(quantlib_delta, abs=1e-10)
    assert vega == pytest.approx(quantlib_vega, abs=1e-10)


HESTON_REFERENCE_CASES = [
    pytest.param(100.0, 100.0, 365, 0.03, 0.04, 0.04, 1.5, 0.4, -0.7, 0.00, id="atm_1y"),
    pytest.param(80.0, 100.0, 90, 0.01, 0.09, 0.04, 2.0, 0.6, -0.5, 0.00, id="otm_3m"),
    pytest.param(
        130.0,
        100.0,
        730,
        0.05,
        0.0225,
        0.09,
        3.0,
        0.8,
        -0.2,
        0.01,
        id="itm_2y_with_dividend",
    ),
    pytest.param(60.0, 100.0, 30, 0.02, 0.25, 0.09, 1.0, 0.8, -0.6, 0.00, id="deep_otm_1m"),
    pytest.param(
        180.0,
        100.0,
        7,
        0.075,
        0.09,
        0.09,
        2.0,
        0.5,
        -0.3,
        0.00,
        id="deep_itm_weekly",
    ),
    pytest.param(
        100.0,
        100.0,
        30,
        0.02,
        0.0009,
        0.0009,
        1.0,
        0.1,
        -0.5,
        0.00,
        id="low_variance_short_maturity",
    ),
    pytest.param(
        70.0,
        100.0,
        730,
        0.04,
        0.16,
        0.36,
        0.7,
        1.2,
        -0.8,
        0.00,
        id="long_high_variance",
    ),
    pytest.param(
        100.0,
        100.0,
        365,
        0.02,
        0.09,
        0.04,
        0.3,
        1.0,
        -0.9,
        0.00,
        id="feller_violating",
    ),
    pytest.param(
        120.0,
        100.0,
        180,
        0.00,
        0.04,
        0.16,
        5.0,
        0.3,
        -0.1,
        0.02,
        id="fast_reversion_with_dividend",
    ),
    pytest.param(
        90.0,
        100.0,
        14,
        0.01,
        0.49,
        0.09,
        0.5,
        1.5,
        -0.85,
        0.00,
        id="short_high_v0_high_vol_of_vol",
    ),
]


HESTON_DELTA_REFERENCE_CASES = [
    pytest.param(100.0, 100.0, 365, 0.03, 0.04, 0.04, 1.5, 0.4, -0.7, 0.00, id="atm_1y"),
    pytest.param(80.0, 100.0, 90, 0.01, 0.09, 0.04, 2.0, 0.6, -0.5, 0.00, id="otm_3m"),
    pytest.param(
        130.0,
        100.0,
        730,
        0.05,
        0.0225,
        0.09,
        3.0,
        0.8,
        -0.2,
        0.01,
        id="itm_2y_with_dividend",
    ),
    pytest.param(60.0, 100.0, 30, 0.02, 0.25, 0.09, 1.0, 0.8, -0.6, 0.00, id="deep_otm_1m"),
    pytest.param(
        100.0,
        100.0,
        30,
        0.02,
        0.0009,
        0.0009,
        1.0,
        0.1,
        -0.5,
        0.00,
        id="low_variance_short_maturity",
    ),
    pytest.param(
        70.0,
        100.0,
        730,
        0.04,
        0.16,
        0.36,
        0.7,
        1.2,
        -0.8,
        0.00,
        id="long_high_variance",
    ),
    pytest.param(
        100.0,
        100.0,
        365,
        0.02,
        0.09,
        0.04,
        0.3,
        1.0,
        -0.9,
        0.00,
        id="feller_violating",
    ),
    pytest.param(
        120.0,
        100.0,
        180,
        0.00,
        0.04,
        0.16,
        5.0,
        0.3,
        -0.1,
        0.02,
        id="fast_reversion_with_dividend",
    ),
    pytest.param(
        90.0,
        100.0,
        14,
        0.01,
        0.49,
        0.09,
        0.5,
        1.5,
        -0.85,
        0.00,
        id="short_high_v0_high_vol_of_vol",
    ),
]


@pytest.mark.parametrize(
    (
        "spot",
        "strike",
        "maturity_days",
        "rate",
        "v0",
        "theta",
        "kappa",
        "xi",
        "rho",
        "dividend_yield",
    ),
    HESTON_REFERENCE_CASES,
)
def test_heston_matches_quantlib_price(
    spot: float,
    strike: float,
    maturity_days: int,
    rate: float,
    v0: float,
    theta: float,
    kappa: float,
    xi: float,
    rho: float,
    dividend_yield: float,
) -> None:
    solver = HestonSolver(absolute_tolerance=1e-9, relative_tolerance=1e-9, quad_limit=500)
    maturity = maturity_days / 365.0

    quantlib_price = quantlib_heston_call_price(
        spot,
        strike,
        maturity_days,
        rate,
        v0,
        theta,
        kappa,
        xi,
        rho,
        dividend_yield,
    )
    price = solver.call_price(
        spot, strike, maturity, rate, v0, theta, kappa, xi, rho, dividend_yield
    )

    assert price == pytest.approx(quantlib_price, abs=5e-6)


@pytest.mark.parametrize(
    (
        "spot",
        "strike",
        "maturity_days",
        "rate",
        "v0",
        "theta",
        "kappa",
        "xi",
        "rho",
        "dividend_yield",
    ),
    HESTON_DELTA_REFERENCE_CASES,
)
def test_heston_delta_matches_quantlib_central_finite_difference(
    spot: float,
    strike: float,
    maturity_days: int,
    rate: float,
    v0: float,
    theta: float,
    kappa: float,
    xi: float,
    rho: float,
    dividend_yield: float,
) -> None:
    solver = HestonSolver(absolute_tolerance=1e-9, relative_tolerance=1e-9, quad_limit=500)
    params = {
        "spot": spot,
        "strike": strike,
        "maturity_days": maturity_days,
        "rate": rate,
        "v0": v0,
        "theta": theta,
        "kappa": kappa,
        "xi": xi,
        "rho": rho,
        "dividend_yield": dividend_yield,
    }
    bump = max(1e-4 * params["spot"], 1e-4)
    up = params | {"spot": params["spot"] + bump}
    down = params | {"spot": params["spot"] - bump}
    quantlib_delta = (
        quantlib_heston_call_price(**up) - quantlib_heston_call_price(**down)
    ) / (2.0 * bump)
    delta = solver.call_delta(
        params["spot"],
        params["strike"],
        params["maturity_days"] / 365.0,
        params["rate"],
        params["v0"],
        params["theta"],
        params["kappa"],
        params["xi"],
        params["rho"],
        params["dividend_yield"],
    )

    assert delta == pytest.approx(quantlib_delta, abs=2e-6)
