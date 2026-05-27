import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.datasets.generator import OptionDataset
from src.evaluation import BinEvaluator, BinPartition, Report
from src.experiments import (
    Experiment,
    ExperimentResult,
    PriceVsIVStudy,
    SurrogateInput,
)
from src.solvers import BlackScholesSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _IdentityPriceModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]


def _build_bs_dataset(rng_seed: int = 7, n_samples: int = 80) -> OptionDataset:
    rng = np.random.default_rng(rng_seed)
    moneyness = rng.uniform(0.6, 1.6, size=n_samples)
    maturity = rng.uniform(0.1, 1.0, size=n_samples)
    rate = rng.uniform(0.0, 0.05, size=n_samples)
    sigma = rng.uniform(0.1, 0.5, size=n_samples)

    bs = BlackScholesSolver()
    prices = np.asarray(
        bs.call_price(
            spot=moneyness, strike=1.0, maturity=maturity,
            rate=rate, volatility=sigma, dividend_yield=0.0,
        ),
        dtype=np.float32,
    )
    deltas = np.asarray(
        bs.call_delta(
            spot=moneyness, strike=1.0, maturity=maturity,
            rate=rate, volatility=sigma, dividend_yield=0.0,
        ),
        dtype=np.float32,
    )

    raw = np.stack([moneyness, maturity, rate, sigma], axis=1).astype(np.float32)
    features = raw.copy()
    features[:, 0] = (features[:, 0] - 0.4) / (2.0 - 0.4)
    features[:, 1] = (features[:, 1] - 7.0 / 365.0) / (2.0 - 7.0 / 365.0)
    features[:, 2] = features[:, 2] / 0.075
    features[:, 3] = (features[:, 3] - 0.03) / (1.0 - 0.03)

    return OptionDataset(
        features=torch.from_numpy(features),
        prices=torch.from_numpy(prices[:, None]),
        deltas=torch.from_numpy(deltas[:, None]),
        raw_inputs=torch.from_numpy(raw),
        input_names=("moneyness", "maturity", "rate", "volatility"),
    )


def _build_bs_input(dataset: OptionDataset | None = None) -> SurrogateInput:
    dataset = dataset if dataset is not None else _build_bs_dataset()
    evaluator = BinEvaluator(
        partition=BinPartition.default(),
        pricer=BlackScholesSolver(),
        device="cpu",
        batch_size=32,
    )
    return SurrogateInput(
        surrogate_id="BS-test",
        model=_IdentityPriceModel(),
        dataset=dataset,
        evaluator=evaluator,
    )


# ---------------------------------------------------------------------------
# Class structure
# ---------------------------------------------------------------------------


def test_experiment_base_class_is_abstract() -> None:
    with pytest.raises(TypeError):
        Experiment()  # type: ignore[abstract]


def test_surrogate_input_is_frozen_dataclass() -> None:
    surrogate_input = _build_bs_input()
    with pytest.raises(Exception):
        surrogate_input.surrogate_id = "other"  # type: ignore[misc]


def test_price_vs_iv_study_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        PriceVsIVStudy(inputs=())


# ---------------------------------------------------------------------------
# run() — single surrogate
# ---------------------------------------------------------------------------


def test_run_returns_experiment_result_with_expected_metadata() -> None:
    study = PriceVsIVStudy(inputs=(_build_bs_input(),))

    result = study.run()

    assert isinstance(result, ExperimentResult)
    assert result.experiment_id == "E1"
    assert result.surrogates == ("BS-test",)
    assert "discrepancia" in result.metric_primary.lower()


def test_run_emits_one_row_per_bin_per_surrogate() -> None:
    study = PriceVsIVStudy(inputs=(_build_bs_input(),))

    result = study.run()

    assert len(result.table) == 25
    bin_ids = sorted(row["bin_id"] for row in result.table)
    assert bin_ids == list(range(25))


def test_table_carries_all_documented_columns() -> None:
    study = PriceVsIVStudy(inputs=(_build_bs_input(),))

    result = study.run()

    expected_columns = {
        "surrogate_id",
        "bin_id",
        "moneyness_idx",
        "maturity_idx",
        "bin_label",
        "n_points",
        "price_mae_mean",
        "price_mae_p95",
        "price_mae_p99",
        "iv_mae_mean",
        "iv_mae_p95",
        "iv_mae_p99",
        "iv_failure_rate",
        "vega_proxy_mean",
        "iv_to_price_ratio",
    }
    assert expected_columns.issubset(set(result.table[0].keys()))


def test_summary_mentions_top_discrepancy_bins() -> None:
    study = PriceVsIVStudy(inputs=(_build_bs_input(),))

    result = study.run()

    assert "E1" in result.summary
    # Each non-empty bin label that actually appears in the table should be
    # a candidate for the summary; at the very least the keyword "ratio"
    # should be present in the top-discrepancy lines.
    assert "ratio" in result.summary.lower() or "iv" in result.summary.lower()


def test_run_stores_reports_keyed_by_surrogate_id() -> None:
    study = PriceVsIVStudy(inputs=(_build_bs_input(),))

    result = study.run()

    assert set(result.reports.keys()) == {"BS-test"}
    assert isinstance(result.reports["BS-test"], Report)
    assert result.reports["BS-test"].iv is not None  # E1 forced compute_iv


# ---------------------------------------------------------------------------
# run() — multiple surrogates
# ---------------------------------------------------------------------------


def test_run_with_two_surrogates_emits_two_blocks() -> None:
    dataset = _build_bs_dataset(rng_seed=11)
    input_a = SurrogateInput(
        surrogate_id="BS-A",
        model=_IdentityPriceModel(),
        dataset=dataset,
        evaluator=BinEvaluator(
            partition=BinPartition.default(),
            pricer=BlackScholesSolver(),
            device="cpu",
            batch_size=32,
        ),
    )
    input_b = SurrogateInput(
        surrogate_id="BS-B",
        model=_IdentityPriceModel(),
        dataset=dataset,
        evaluator=BinEvaluator(
            partition=BinPartition.default(),
            pricer=BlackScholesSolver(),
            device="cpu",
            batch_size=32,
        ),
    )
    study = PriceVsIVStudy(inputs=(input_a, input_b))

    result = study.run()

    assert result.surrogates == ("BS-A", "BS-B")
    assert len(result.table) == 50  # 25 bins x 2 surrogates
    ids_in_table = {row["surrogate_id"] for row in result.table}
    assert ids_in_table == {"BS-A", "BS-B"}


# ---------------------------------------------------------------------------
# Vega proxy
# ---------------------------------------------------------------------------


def test_vega_proxy_is_finite_where_iv_inversion_succeeds() -> None:
    study = PriceVsIVStudy(inputs=(_build_bs_input(),))

    result = study.run()

    # On a 80-point synthetic BS dataset, IV inversion is essentially
    # guaranteed to succeed in most bins; the Vega proxy column must
    # therefore contain some finite values.
    finite_vegas = [
        row["vega_proxy_mean"]
        for row in result.table
        if np.isfinite(row["vega_proxy_mean"])
    ]
    assert finite_vegas, "expected at least one finite Vega proxy entry"


def test_iv_to_price_ratio_is_finite_in_non_trivial_bins() -> None:
    study = PriceVsIVStudy(inputs=(_build_bs_input(),))

    result = study.run()

    finite_ratios = [
        row["iv_to_price_ratio"]
        for row in result.table
        if np.isfinite(row["iv_to_price_ratio"])
    ]
    assert finite_ratios
    assert all(value >= 0.0 for value in finite_ratios)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_result_to_csv_writes_one_header_plus_rows(tmp_path: Path) -> None:
    study = PriceVsIVStudy(inputs=(_build_bs_input(),))
    result = study.run()
    output = tmp_path / "e1.csv"

    result.to_csv(output)

    rows = list(csv.reader(output.open(encoding="utf-8")))
    assert len(rows) == 26  # 1 header + 25 bins


def test_result_to_csv_round_trips_columns(tmp_path: Path) -> None:
    study = PriceVsIVStudy(inputs=(_build_bs_input(),))
    result = study.run()
    output = tmp_path / "e1.csv"

    result.to_csv(output)

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert rows
    expected = set(result.table[0].keys())
    assert expected.issubset(set(rows[0].keys()))


def test_result_to_heatmaps_writes_two_pngs_per_surrogate(tmp_path: Path) -> None:
    study = PriceVsIVStudy(inputs=(_build_bs_input(),))
    result = study.run()

    written = result.to_heatmaps(tmp_path)

    # By default: 2 metrics (price + iv) x 1 surrogate = 2 PNGs.
    assert len(written) == 2
    for path in written:
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0
