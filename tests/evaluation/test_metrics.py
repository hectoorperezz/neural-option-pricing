import numpy as np
import pytest
import torch
from torch import nn

from src.evaluation.metrics import (
    absolute_errors,
    aggregate_by_bin,
    invert_implied_volatility_call,
    predict_surrogate_prices_and_deltas,
)
from src.solvers import BlackScholesSolver
from src.solvers.iv import ImpliedVolatilityInverter


# --- absolute_errors -------------------------------------------------------


def test_absolute_errors_on_numpy_arrays() -> None:
    predicted = np.array([1.0, 2.0, 3.0])
    target = np.array([1.5, 1.5, 2.0])

    errors = absolute_errors(predicted, target)

    np.testing.assert_allclose(errors, [0.5, 0.5, 1.0])


def test_absolute_errors_on_torch_tensors() -> None:
    predicted = torch.tensor([1.0, 2.0, -1.0])
    target = torch.tensor([1.5, 0.5, 1.0])

    errors = absolute_errors(predicted, target)

    assert isinstance(errors, np.ndarray)
    np.testing.assert_allclose(errors, [0.5, 1.5, 2.0])


def test_absolute_errors_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        absolute_errors(np.array([1.0, 2.0]), np.array([1.0]))


# --- aggregate_by_bin ------------------------------------------------------


def test_aggregate_single_bin_matches_numpy() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    bin_id = np.zeros(4, dtype=np.int64)

    result = aggregate_by_bin(values, bin_id, n_bins=1)

    assert int(result["count"][0]) == 4
    np.testing.assert_allclose(result["mean"][0], values.mean())
    np.testing.assert_allclose(result["p50"][0], np.percentile(values, 50))
    np.testing.assert_allclose(result["p95"][0], np.percentile(values, 95))
    np.testing.assert_allclose(result["p99"][0], np.percentile(values, 99))


def test_aggregate_two_bins_split_correctly() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    bin_id = np.array([0, 0, 1, 1])

    result = aggregate_by_bin(values, bin_id, n_bins=2)

    assert result["count"].tolist() == [2, 2]
    np.testing.assert_allclose(result["mean"], [1.5, 3.5])


def test_aggregate_empty_bin_returns_nan_and_zero_count() -> None:
    values = np.array([1.0, 2.0])
    bin_id = np.array([0, 0])

    result = aggregate_by_bin(values, bin_id, n_bins=3)

    assert int(result["count"][2]) == 0
    assert np.isnan(result["mean"][2])
    assert np.isnan(result["p50"][2])
    assert np.isnan(result["p95"][2])


def test_aggregate_ignores_nan_values() -> None:
    values = np.array([1.0, np.nan, 3.0])
    bin_id = np.array([0, 0, 0])

    result = aggregate_by_bin(values, bin_id, n_bins=1)

    # NaN is excluded from both count and aggregates
    assert int(result["count"][0]) == 2
    np.testing.assert_allclose(result["mean"][0], 2.0)


def test_aggregate_all_nan_in_bin_returns_zero_count_and_nan_aggregates() -> None:
    values = np.array([np.nan, np.nan, 1.0])
    bin_id = np.array([0, 0, 1])

    result = aggregate_by_bin(values, bin_id, n_bins=2)

    assert int(result["count"][0]) == 0
    assert np.isnan(result["mean"][0])
    assert int(result["count"][1]) == 1
    np.testing.assert_allclose(result["mean"][1], 1.0)


def test_aggregate_custom_percentiles() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bin_id = np.zeros(5, dtype=np.int64)

    result = aggregate_by_bin(values, bin_id, n_bins=1, percentiles=(25, 75))

    assert "p25" in result
    assert "p75" in result
    assert "p50" not in result
    np.testing.assert_allclose(result["p25"][0], np.percentile(values, 25))


def test_aggregate_rejects_out_of_range_bin_id() -> None:
    values = np.array([1.0, 2.0])
    bin_id = np.array([0, 5])

    with pytest.raises(ValueError, match="bin_id"):
        aggregate_by_bin(values, bin_id, n_bins=3)


def test_aggregate_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        aggregate_by_bin(np.array([1.0, 2.0]), np.array([0]), n_bins=1)


# --- predict_surrogate_prices_and_deltas -----------------------------------


class _TinyMLP(nn.Module):
    def __init__(self, input_dim: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_predict_returns_correct_shapes() -> None:
    torch.manual_seed(0)
    model = _TinyMLP()
    features = np.random.default_rng(0).uniform(0.0, 1.0, size=(50, 4)).astype(np.float32)

    prices, deltas = predict_surrogate_prices_and_deltas(
        model,
        features,
        batch_size=16,
        device="cpu",
    )

    assert prices.shape == (50,)
    assert deltas.shape == (50,)


def test_predict_is_invariant_to_batch_size() -> None:
    torch.manual_seed(0)
    model = _TinyMLP()
    features = torch.randn(80, 4)

    prices_small, deltas_small = predict_surrogate_prices_and_deltas(
        model, features, batch_size=8, device="cpu"
    )
    prices_large, deltas_large = predict_surrogate_prices_and_deltas(
        model, features, batch_size=80, device="cpu"
    )

    np.testing.assert_allclose(prices_small, prices_large, atol=1e-6)
    np.testing.assert_allclose(deltas_small, deltas_large, atol=1e-6)


def test_predict_delta_matches_manual_chain_rule_on_linear_model() -> None:
    # For a linear model y = w0*m_norm + w1*x1 + w2*x2 + w3*x3 + b,
    # dy/dm = w0 / (m_max - m_min).
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


def test_predict_rejects_bad_arguments() -> None:
    model = _TinyMLP()
    features = torch.zeros(4, 4)

    with pytest.raises(ValueError, match="batch_size"):
        predict_surrogate_prices_and_deltas(model, features, batch_size=0)
    with pytest.raises(ValueError, match="moneyness_range"):
        predict_surrogate_prices_and_deltas(
            model, features, moneyness_range=(2.0, 0.4)
        )


# --- invert_implied_volatility_call ---------------------------------------


def test_invert_iv_roundtrip_on_black_scholes() -> None:
    bs = BlackScholesSolver()
    inverter = ImpliedVolatilityInverter(solver=bs)
    sigma_true = 0.3
    moneyness = 1.0
    maturity = 0.5
    rate = 0.02

    price_true = float(
        bs.call_price(
            spot=moneyness,
            strike=1.0,
            maturity=maturity,
            rate=rate,
            volatility=sigma_true,
            dividend_yield=0.0,
        )
    )

    iv, ok = invert_implied_volatility_call(
        prices=np.array([price_true]),
        moneyness=np.array([moneyness]),
        maturity=np.array([maturity]),
        rate=np.array([rate]),
        inverter=inverter,
    )

    assert bool(ok[0])
    np.testing.assert_allclose(iv[0], sigma_true, atol=1e-6)


def test_invert_iv_marks_negative_price_as_failure() -> None:
    iv, ok = invert_implied_volatility_call(
        prices=np.array([-0.1]),
        moneyness=np.array([1.0]),
        maturity=np.array([0.5]),
        rate=np.array([0.02]),
    )

    assert not bool(ok[0])
    assert np.isnan(iv[0])


@pytest.mark.parametrize(
    ("prices", "moneyness", "maturity", "rate"),
    [
        ([np.nan], [1.0], [0.5], [0.02]),
        ([0.1], [np.nan], [0.5], [0.02]),
        ([0.1], [1.0], [np.nan], [0.02]),
        ([0.1], [1.0], [0.5], [np.nan]),
    ],
)
def test_invert_iv_marks_non_finite_inputs_as_failure(
    prices: list[float],
    moneyness: list[float],
    maturity: list[float],
    rate: list[float],
) -> None:
    iv, ok = invert_implied_volatility_call(
        prices=np.array(prices),
        moneyness=np.array(moneyness),
        maturity=np.array(maturity),
        rate=np.array(rate),
    )

    assert not bool(ok[0])
    assert np.isnan(iv[0])


def test_invert_iv_marks_above_intrinsic_as_failure() -> None:
    # A call cannot be worth more than the discounted spot (strike=1, q=0).
    iv, ok = invert_implied_volatility_call(
        prices=np.array([10.0]),
        moneyness=np.array([1.0]),
        maturity=np.array([0.5]),
        rate=np.array([0.02]),
    )

    assert not bool(ok[0])
    assert np.isnan(iv[0])


def test_invert_iv_vectorised_round_trip() -> None:
    bs = BlackScholesSolver()
    inverter = ImpliedVolatilityInverter(solver=bs)
    rng = np.random.default_rng(123)
    n = 25
    sigma_true = rng.uniform(0.05, 0.6, size=n)
    moneyness = rng.uniform(0.8, 1.3, size=n)
    maturity = rng.uniform(0.1, 1.5, size=n)
    rate = rng.uniform(0.0, 0.05, size=n)

    prices = np.array(
        [
            float(
                bs.call_price(
                    spot=float(moneyness[i]),
                    strike=1.0,
                    maturity=float(maturity[i]),
                    rate=float(rate[i]),
                    volatility=float(sigma_true[i]),
                    dividend_yield=0.0,
                )
            )
            for i in range(n)
        ]
    )

    iv, ok = invert_implied_volatility_call(
        prices=prices,
        moneyness=moneyness,
        maturity=maturity,
        rate=rate,
        inverter=inverter,
    )

    assert bool(ok.all())
    np.testing.assert_allclose(iv, sigma_true, atol=1e-5)


def test_invert_iv_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        invert_implied_volatility_call(
            prices=np.array([0.1, 0.2]),
            moneyness=np.array([1.0]),
            maturity=np.array([0.5, 1.0]),
            rate=np.array([0.02, 0.02]),
        )
