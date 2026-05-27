import numpy as np
import pytest
import torch
from torch import nn

from src.datasets.generator import OptionDataset
from src.evaluation import BinEvaluator, BinPartition, Report
from src.solvers import BlackScholesSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _IdentityPriceModel(nn.Module):
    """A model whose first feature is its prediction. Useful for sanity tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]


def _build_bs_test_dataset(rng_seed: int = 42, n_samples: int = 200) -> OptionDataset:
    """Build a small in-domain BS test set with reference prices and deltas."""
    rng = np.random.default_rng(rng_seed)
    moneyness = rng.uniform(0.5, 1.8, size=n_samples)
    maturity = rng.uniform(7.0 / 365.0, 1.5, size=n_samples)
    rate = rng.uniform(0.0, 0.07, size=n_samples)
    sigma = rng.uniform(0.05, 0.6, size=n_samples)

    bs = BlackScholesSolver()
    prices = bs.call_price(
        spot=moneyness,
        strike=1.0,
        maturity=maturity,
        rate=rate,
        volatility=sigma,
        dividend_yield=0.0,
    )
    deltas = bs.call_delta(
        spot=moneyness,
        strike=1.0,
        maturity=maturity,
        rate=rate,
        volatility=sigma,
        dividend_yield=0.0,
    )

    raw = np.stack([moneyness, maturity, rate, sigma], axis=1).astype(np.float32)
    # Normalize features to [0, 1] using the same ranges the project uses.
    features = raw.copy()
    features[:, 0] = (features[:, 0] - 0.4) / (2.0 - 0.4)
    features[:, 1] = (features[:, 1] - 7.0 / 365.0) / (2.0 - 7.0 / 365.0)
    features[:, 2] = features[:, 2] / 0.075
    features[:, 3] = (features[:, 3] - 0.03) / (1.0 - 0.03)

    return OptionDataset(
        features=torch.from_numpy(features),
        prices=torch.from_numpy(np.asarray(prices, dtype=np.float32)[:, None]),
        deltas=torch.from_numpy(np.asarray(deltas, dtype=np.float32)[:, None]),
        raw_inputs=torch.from_numpy(raw),
        input_names=("moneyness", "maturity", "rate", "volatility"),
    )


def _build_evaluator() -> BinEvaluator:
    return BinEvaluator(
        partition=BinPartition.default(),
        pricer=BlackScholesSolver(),
        device="cpu",
        batch_size=64,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_constructor_stores_partition_and_pricer() -> None:
    partition = BinPartition.default()
    bs = BlackScholesSolver()

    evaluator = BinEvaluator(partition=partition, pricer=bs)

    assert evaluator.partition is partition
    assert evaluator.pricer is bs
    assert evaluator.iv_inverter is None  # default, resolved at evaluate-time


# ---------------------------------------------------------------------------
# evaluate() — basic structure
# ---------------------------------------------------------------------------


def test_evaluate_returns_report_with_expected_metadata() -> None:
    dataset = _build_bs_test_dataset()
    evaluator = _build_evaluator()
    model = _IdentityPriceModel()

    report = evaluator.evaluate(
        surrogate=model,
        dataset=dataset,
        compute_iv=False,
        surrogate_id="dummy",
        test_path="(synthetic)",
    )

    assert isinstance(report, Report)
    assert report.surrogate_id == "dummy"
    assert report.test_path == "(synthetic)"
    assert report.n_samples == 200
    assert report.partition is evaluator.partition


def test_evaluate_populates_price_aggregates_with_correct_shape() -> None:
    dataset = _build_bs_test_dataset()
    evaluator = _build_evaluator()

    report = evaluator.evaluate(
        surrogate=_IdentityPriceModel(),
        dataset=dataset,
        compute_iv=False,
    )

    n_bins = evaluator.partition.n_bins
    for key in ("mean", "count", "p50", "p95", "p99"):
        assert report.price[key].shape == (n_bins,)


def test_evaluate_skips_iv_when_compute_iv_is_false() -> None:
    dataset = _build_bs_test_dataset()
    evaluator = _build_evaluator()

    report = evaluator.evaluate(
        surrogate=_IdentityPriceModel(),
        dataset=dataset,
        compute_iv=False,
    )

    assert report.iv is None
    assert report.iv_failure_rate_per_bin is None


def test_evaluate_returns_delta_none_when_dataset_has_no_deltas() -> None:
    dataset_with = _build_bs_test_dataset()
    dataset = OptionDataset(
        features=dataset_with.features,
        prices=dataset_with.prices,
        deltas=None,
        raw_inputs=dataset_with.raw_inputs,
        input_names=dataset_with.input_names,
    )
    evaluator = _build_evaluator()

    report = evaluator.evaluate(
        surrogate=_IdentityPriceModel(),
        dataset=dataset,
        compute_iv=False,
    )

    assert report.delta is None


def test_evaluate_populates_delta_aggregates_when_dataset_has_deltas() -> None:
    dataset = _build_bs_test_dataset()
    evaluator = _build_evaluator()

    report = evaluator.evaluate(
        surrogate=_IdentityPriceModel(),
        dataset=dataset,
        compute_iv=False,
    )

    assert report.delta is not None
    n_bins = evaluator.partition.n_bins
    assert report.delta["mean"].shape == (n_bins,)


# ---------------------------------------------------------------------------
# evaluate() — numerical correctness
# ---------------------------------------------------------------------------


def test_price_mae_per_bin_matches_manual_calculation() -> None:
    dataset = _build_bs_test_dataset()
    evaluator = _build_evaluator()
    model = _IdentityPriceModel()

    report = evaluator.evaluate(
        surrogate=model,
        dataset=dataset,
        compute_iv=False,
    )

    # Reproduce the calculation by hand
    moneyness_feature = dataset.features[:, 0].numpy()
    target_prices = dataset.prices.numpy().reshape(-1)
    manual_errors = np.abs(moneyness_feature - target_prices)
    bin_id, _, _ = evaluator.partition.assign(
        dataset.raw_inputs[:, 0].numpy(),
        dataset.raw_inputs[:, 1].numpy(),
    )
    for k in range(evaluator.partition.n_bins):
        mask = bin_id == k
        if not mask.any():
            assert np.isnan(report.price["mean"][k])
        else:
            np.testing.assert_allclose(
                report.price["mean"][k],
                float(manual_errors[mask].mean()),
                atol=1e-6,
            )


# ---------------------------------------------------------------------------
# evaluate() — bin_id handling
# ---------------------------------------------------------------------------


def test_evaluate_uses_provided_bin_id() -> None:
    dataset = _build_bs_test_dataset()
    evaluator = _build_evaluator()

    # Force every point into bin 0 to make the contract observable
    forced_bin_id = np.zeros(dataset.features.shape[0], dtype=np.int64)
    report = evaluator.evaluate(
        surrogate=_IdentityPriceModel(),
        dataset=dataset,
        bin_id=forced_bin_id,
        compute_iv=False,
    )

    assert int(report.price["count"][0]) == dataset.features.shape[0]
    for k in range(1, evaluator.partition.n_bins):
        assert int(report.price["count"][k]) == 0


def test_evaluate_rejects_bin_id_with_wrong_shape() -> None:
    dataset = _build_bs_test_dataset()
    evaluator = _build_evaluator()

    with pytest.raises(ValueError, match="bin_id"):
        evaluator.evaluate(
            surrogate=_IdentityPriceModel(),
            dataset=dataset,
            bin_id=np.zeros(5, dtype=np.int64),
            compute_iv=False,
        )


def test_evaluate_rejects_bin_id_out_of_range() -> None:
    dataset = _build_bs_test_dataset()
    evaluator = _build_evaluator()

    bin_id = np.full(dataset.features.shape[0], 999, dtype=np.int64)
    with pytest.raises(ValueError, match="bin_id"):
        evaluator.evaluate(
            surrogate=_IdentityPriceModel(),
            dataset=dataset,
            bin_id=bin_id,
            compute_iv=False,
        )


def test_resolve_bin_id_computed_on_the_fly_matches_partition_assign() -> None:
    dataset = _build_bs_test_dataset()
    evaluator = _build_evaluator()

    auto_report = evaluator.evaluate(
        surrogate=_IdentityPriceModel(),
        dataset=dataset,
        compute_iv=False,
    )
    manual_bin_id, _, _ = evaluator.partition.assign(
        dataset.raw_inputs[:, 0].numpy(), dataset.raw_inputs[:, 1].numpy()
    )
    explicit_report = evaluator.evaluate(
        surrogate=_IdentityPriceModel(),
        dataset=dataset,
        bin_id=manual_bin_id,
        compute_iv=False,
    )

    np.testing.assert_allclose(
        auto_report.price["mean"], explicit_report.price["mean"], equal_nan=True
    )


# ---------------------------------------------------------------------------
# evaluate() — IV computation
# ---------------------------------------------------------------------------


def test_evaluate_with_iv_returns_aggregates_and_failure_rate() -> None:
    dataset = _build_bs_test_dataset(n_samples=80)
    evaluator = _build_evaluator()

    report = evaluator.evaluate(
        surrogate=_IdentityPriceModel(),
        dataset=dataset,
        compute_iv=True,
    )

    assert report.iv is not None
    assert report.iv["mean"].shape == (evaluator.partition.n_bins,)
    assert report.iv_failure_rate_per_bin is not None
    assert report.iv_failure_rate_per_bin.shape == (evaluator.partition.n_bins,)
    # Failure rate must lie in [0, 1] wherever it is finite
    finite_rates = report.iv_failure_rate_per_bin[
        np.isfinite(report.iv_failure_rate_per_bin)
    ]
    assert np.all((finite_rates >= 0.0) & (finite_rates <= 1.0))


# ---------------------------------------------------------------------------
# rawinputs validation
# ---------------------------------------------------------------------------


def test_evaluate_rejects_raw_inputs_with_too_few_columns() -> None:
    n = 10
    features = torch.zeros(n, 2)
    prices = torch.zeros(n, 1)
    raw = torch.zeros(n, 2)  # only 2 columns, needs at least 3
    dataset = OptionDataset(
        features=features,
        prices=prices,
        deltas=None,
        raw_inputs=raw,
        input_names=("moneyness", "maturity"),
    )
    evaluator = _build_evaluator()

    with pytest.raises(ValueError, match="raw_inputs"):
        evaluator.evaluate(
            surrogate=_IdentityPriceModel(),
            dataset=dataset,
            compute_iv=False,
        )
