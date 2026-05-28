import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.datasets.generator import OptionDataset
from src.evaluation import BinEvaluator, BinPartition, Report
from src.experiments import (
    ActivationStudy,
    Experiment,
    ExperimentResult,
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


def _make_input(
    surrogate_id: str,
    activation: str | None = None,
    dataset: OptionDataset | None = None,
) -> SurrogateInput:
    dataset = dataset if dataset is not None else _build_bs_dataset()
    evaluator = BinEvaluator(
        partition=BinPartition.default(),
        pricer=BlackScholesSolver(),
        device="cpu",
        batch_size=32,
    )
    return SurrogateInput(
        surrogate_id=surrogate_id,
        model=_IdentityPriceModel(),
        dataset=dataset,
        evaluator=evaluator,
        labels={"activation": activation} if activation is not None else None,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_activation_study_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ActivationStudy(inputs=())


def test_activation_study_is_experiment_subclass() -> None:
    study = ActivationStudy(inputs=(_make_input("BS-3", activation="swish"),))
    assert isinstance(study, Experiment)


# ---------------------------------------------------------------------------
# run() — structure
# ---------------------------------------------------------------------------


def test_run_returns_experiment_result_with_expected_metadata() -> None:
    study = ActivationStudy(
        inputs=(_make_input("BS-3", activation="swish"),),
        family_label="Black-Scholes",
    )

    result = study.run()

    assert isinstance(result, ExperimentResult)
    assert result.experiment_id == "E2"
    assert result.surrogates == ("BS-3",)
    assert "MAE_Delta" in result.metric_primary


def test_run_emits_one_row_per_bin_per_surrogate() -> None:
    study = ActivationStudy(
        inputs=(
            _make_input("BS-1", activation="relu"),
            _make_input("BS-3", activation="swish"),
        ),
    )

    result = study.run()

    assert len(result.table) == 50  # 2 surrogates x 25 bins


def test_table_carries_all_expected_columns() -> None:
    study = ActivationStudy(inputs=(_make_input("BS-3", activation="swish"),))

    result = study.run()

    expected = {
        "surrogate_id",
        "activation",
        "bin_id",
        "moneyness_idx",
        "maturity_idx",
        "bin_label",
        "n_points",
        "price_mae_mean",
        "price_mae_p95",
        "price_mae_p99",
        "delta_mae_mean",
        "delta_mae_p95",
        "delta_mae_p99",
    }
    assert expected.issubset(set(result.table[0].keys()))


def test_activation_column_is_populated_from_labels() -> None:
    study = ActivationStudy(
        inputs=(
            _make_input("BS-1", activation="relu"),
            _make_input("BS-3", activation="swish"),
        ),
    )
    result = study.run()

    activations_by_surrogate = {
        row["surrogate_id"]: row["activation"] for row in result.table
    }
    assert activations_by_surrogate["BS-1"] == "relu"
    assert activations_by_surrogate["BS-3"] == "swish"


def test_missing_labels_yields_empty_activation_string() -> None:
    study = ActivationStudy(inputs=(_make_input("BS-3", activation=None),))
    result = study.run()

    assert result.table[0]["activation"] == ""


def test_dataset_without_deltas_raises_clearly() -> None:
    dataset_with_deltas = _build_bs_dataset()
    dataset_without_deltas = OptionDataset(
        features=dataset_with_deltas.features,
        prices=dataset_with_deltas.prices,
        deltas=None,
        raw_inputs=dataset_with_deltas.raw_inputs,
        input_names=dataset_with_deltas.input_names,
    )
    study = ActivationStudy(
        inputs=(_make_input("BS-3", activation="swish", dataset=dataset_without_deltas),)
    )

    with pytest.raises(RuntimeError, match="MAE_Delta"):
        study.run()


# ---------------------------------------------------------------------------
# run() — semantics: compute_iv is forced off (no IV columns)
# ---------------------------------------------------------------------------


def test_reports_have_no_iv_aggregates() -> None:
    study = ActivationStudy(inputs=(_make_input("BS-3", activation="swish"),))
    result = study.run()

    report = result.reports["BS-3"]
    assert report.iv is None
    assert report.iv_failure_rate_per_bin is None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_mentions_ranking_by_activation() -> None:
    study = ActivationStudy(
        inputs=(
            _make_input("BS-1", activation="relu"),
            _make_input("BS-3", activation="swish"),
        ),
        family_label="Black-Scholes",
    )
    result = study.run()

    summary = result.summary
    assert "Black-Scholes" in summary
    assert "MAE_Delta" in summary
    assert "relu" in summary.lower() or "swish" in summary.lower()


def test_summary_states_no_strong_weak_classification() -> None:
    study = ActivationStudy(inputs=(_make_input("BS-3", activation="swish"),))
    result = study.run()

    # The methodology document forbids a fuerte/débil verdict for E2.
    assert "observacional" in result.summary.lower()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_to_csv_writes_header_plus_rows(tmp_path: Path) -> None:
    study = ActivationStudy(
        inputs=(
            _make_input("BS-1", activation="relu"),
            _make_input("BS-3", activation="swish"),
        ),
    )
    result = study.run()
    output = tmp_path / "e2.csv"

    result.to_csv(output)

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert len(rows) == 50
    assert rows[0]["activation"] in ("relu", "swish")


def test_to_heatmaps_writes_price_and_delta_per_surrogate(tmp_path: Path) -> None:
    study = ActivationStudy(
        inputs=(
            _make_input("BS-1", activation="relu"),
            _make_input("BS-3", activation="swish"),
        ),
    )
    result = study.run()

    written = result.to_heatmaps(tmp_path, metrics=("price", "delta"))

    # 2 surrogates x 2 metrics = 4 PNGs
    assert len(written) == 4
    for path in written:
        assert path.exists()
        assert path.suffix == ".png"
