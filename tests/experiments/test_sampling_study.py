import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.datasets.generator import OptionDataset
from src.evaluation import BinEvaluator, BinPartition
from src.experiments import (
    Experiment,
    ExperimentResult,
    SamplingStudy,
    SurrogateInput,
)
from src.experiments.sampling_study import _relative_degradation, decide_verdict
from src.solvers import BlackScholesSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _LinearPriceModel(nn.Module):
    """Returns ``x[:, :1] * scale``, linear in moneyness so autograd works."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self._scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1] * self._scale


def _build_bs_dataset(rng_seed: int = 11, n_samples: int = 120) -> OptionDataset:
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
    sampler: str | None,
    *,
    model: nn.Module | None = None,
    dataset: OptionDataset | None = None,
) -> SurrogateInput:
    dataset = dataset if dataset is not None else _build_bs_dataset()
    evaluator = BinEvaluator(
        partition=BinPartition.default(),
        pricer=BlackScholesSolver(),
        device="cpu",
        batch_size=64,
    )
    labels = {"sampler": sampler} if sampler is not None else None
    return SurrogateInput(
        surrogate_id=surrogate_id,
        model=model if model is not None else _LinearPriceModel(),
        dataset=dataset,
        evaluator=evaluator,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_rejects_wrong_number_of_inputs() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        SamplingStudy(inputs=(_make_input("H-3", "uniform"),))


def test_rejects_inputs_with_same_sampler() -> None:
    with pytest.raises(ValueError, match="distinct samplers|expects one input"):
        SamplingStudy(
            inputs=(
                _make_input("H-3", "uniform"),
                _make_input("H-5", "uniform"),
            )
        )


def test_rejects_missing_sampler_label() -> None:
    with pytest.raises(ValueError, match="sampler"):
        SamplingStudy(
            inputs=(
                _make_input("H-3", "uniform"),
                _make_input("H-5", None),
            )
        )


def test_is_experiment_subclass() -> None:
    study = SamplingStudy(
        inputs=(
            _make_input("H-3", "uniform"),
            _make_input("H-5", "focused"),
        )
    )
    assert isinstance(study, Experiment)


# ---------------------------------------------------------------------------
# run() — structure
# ---------------------------------------------------------------------------


def _run_dual(
    uniform_scale: float = 1.0,
    focused_scale: float = 1.0,
) -> ExperimentResult:
    dataset = _build_bs_dataset()
    study = SamplingStudy(
        inputs=(
            _make_input(
                "H-3",
                "uniform",
                model=_LinearPriceModel(uniform_scale),
                dataset=dataset,
            ),
            _make_input(
                "H-5",
                "focused",
                model=_LinearPriceModel(focused_scale),
                dataset=dataset,
            ),
        )
    )
    return study.run()


def test_run_returns_experiment_result_with_expected_metadata() -> None:
    result = _run_dual()
    assert isinstance(result, ExperimentResult)
    assert result.experiment_id == "E3"
    assert set(result.surrogates) == {"H-3", "H-5"}
    assert "MAE_IV" in result.metric_primary


def test_run_emits_one_row_per_bin_per_surrogate() -> None:
    result = _run_dual()
    assert len(result.table) == 50  # 2 surrogates x 25 bins


def test_table_carries_expected_columns() -> None:
    result = _run_dual()
    expected = {
        "surrogate_id",
        "sampler",
        "bin_id",
        "moneyness_idx",
        "maturity_idx",
        "bin_label",
        "is_critical",
        "n_points",
        "price_mae_mean",
        "price_mae_p95",
        "price_mae_p99",
        "iv_mae_mean",
        "iv_mae_p95",
        "iv_mae_p99",
        "iv_failure_rate",
    }
    assert expected.issubset(set(result.table[0].keys()))


def test_sampler_column_populated_from_labels() -> None:
    result = _run_dual()
    samplers_by_surrogate = {
        row["surrogate_id"]: row["sampler"] for row in result.table
    }
    assert samplers_by_surrogate["H-3"] == "uniform"
    assert samplers_by_surrogate["H-5"] == "focused"


def test_critical_flag_marks_atm_weekly_short_medium_short() -> None:
    result = _run_dual()
    critical_labels = {
        row["bin_label"] for row in result.table if row["is_critical"]
    }
    assert critical_labels == {"atm_weekly", "atm_short", "atm_medium_short"}


def test_reports_include_iv_aggregates_because_compute_iv_is_forced() -> None:
    result = _run_dual()
    for report in result.reports.values():
        assert report.iv is not None
        assert report.iv_failure_rate_per_bin is not None


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def test_decide_verdict_strong() -> None:
    assert decide_verdict(0.15, 0.05) == "positivo_fuerte"


def test_decide_verdict_weak_low_improvement() -> None:
    assert decide_verdict(0.05, 0.0) == "positivo_debil"


def test_decide_verdict_weak_moderate_global_degradation() -> None:
    # >10% global degradation but <=20%; improvement is positive => weak
    assert decide_verdict(0.15, 0.15) == "positivo_debil"


def test_decide_verdict_negative_no_improvement() -> None:
    assert decide_verdict(0.0, 0.0) == "negativo"
    assert decide_verdict(-0.05, 0.0) == "negativo"


def test_decide_verdict_negative_excessive_global() -> None:
    assert decide_verdict(0.30, 0.25) == "negativo"


def test_global_degradation_uses_uniform_baseline_denominator() -> None:
    uniform_global = 0.16470
    focused_global = 0.12973

    assert _relative_degradation(uniform_global, focused_global) == pytest.approx(
        -0.2123,
        abs=1e-4,
    )


def test_run_populates_verdict_field() -> None:
    result = _run_dual()
    assert result.verdict in {"positivo_fuerte", "positivo_debil", "negativo"}


def test_run_emits_negative_verdict_when_focused_is_clearly_worse() -> None:
    # uniform predicts ~target slope; focused triples it, blowing up IV everywhere.
    result = _run_dual(uniform_scale=1.0, focused_scale=3.0)
    assert result.verdict == "negativo"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_mentions_critical_bins_and_verdict() -> None:
    result = _run_dual()
    summary = result.summary
    assert "atm_weekly" in summary
    assert "atm_short" in summary
    assert "atm_medium_short" in summary
    assert "Veredicto" in summary


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_to_csv_writes_header_plus_rows(tmp_path: Path) -> None:
    result = _run_dual()
    output = tmp_path / "e3.csv"
    result.to_csv(output)

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert len(rows) == 50
    samplers = {row["sampler"] for row in rows}
    assert samplers == {"uniform", "focused"}


def test_to_heatmaps_writes_price_and_iv_per_surrogate(tmp_path: Path) -> None:
    result = _run_dual()
    written = result.to_heatmaps(tmp_path, metrics=("price", "iv"))
    # 2 surrogates x 2 metrics = 4 PNGs
    assert len(written) == 4
    for path in written:
        assert path.exists()
        assert path.suffix == ".png"
