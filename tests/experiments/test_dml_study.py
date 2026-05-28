import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.datasets.generator import OptionDataset
from src.evaluation import BinEvaluator, BinPartition
from src.experiments import (
    DMLStudy,
    Experiment,
    ExperimentResult,
    SurrogateInput,
)
from src.experiments.dml_study import decide_verdict
from src.solvers import BlackScholesSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _LinearPriceModel(nn.Module):
    """Linear-in-features model; autograd-compatible for Delta extraction."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self._scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1] * self._scale


def _build_bs_dataset(rng_seed: int = 17, n_samples: int = 120) -> OptionDataset:
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
    role: str | None,
    *,
    model: nn.Module | None = None,
    dataset: OptionDataset | None = None,
    loss: str = "",
) -> SurrogateInput:
    dataset = dataset if dataset is not None else _build_bs_dataset()
    evaluator = BinEvaluator(
        partition=BinPartition.default(),
        pricer=BlackScholesSolver(),
        device="cpu",
        batch_size=64,
    )
    labels: dict[str, str] | None
    if role is None:
        labels = None
    else:
        labels = {"role": role, "loss": loss}
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


def test_rejects_single_input() -> None:
    with pytest.raises(ValueError, match="DMLStudy compares"):
        DMLStudy(inputs=(_make_input("H-3-small", "small_price"),))


def test_rejects_four_inputs() -> None:
    with pytest.raises(ValueError, match="DMLStudy compares"):
        DMLStudy(
            inputs=(
                _make_input("a", "small_price"),
                _make_input("b", "small_dml"),
                _make_input("c", "baseline_large"),
                _make_input("d", "small_price"),
            )
        )


def test_rejects_missing_small_price_role() -> None:
    with pytest.raises(ValueError, match="small_price"):
        DMLStudy(
            inputs=(
                _make_input("H-6-small", "small_dml"),
                _make_input("H-3", "baseline_large"),
            )
        )


def test_rejects_missing_small_dml_role() -> None:
    with pytest.raises(ValueError, match="small_dml"):
        DMLStudy(
            inputs=(
                _make_input("H-3-small", "small_price"),
                _make_input("H-3", "baseline_large"),
            )
        )


def test_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="unsupported role"):
        DMLStudy(
            inputs=(
                _make_input("a", "small_price"),
                _make_input("b", "small_dml"),
                _make_input("c", "weird_role"),
            )
        )


def test_rejects_duplicate_role() -> None:
    with pytest.raises(ValueError, match="at most once"):
        DMLStudy(
            inputs=(
                _make_input("a", "small_price"),
                _make_input("b", "small_dml"),
                _make_input("c", "small_dml"),
            )
        )


def test_is_experiment_subclass() -> None:
    study = DMLStudy(
        inputs=(
            _make_input("H-3-small", "small_price"),
            _make_input("H-6-small", "small_dml"),
        )
    )
    assert isinstance(study, Experiment)


# ---------------------------------------------------------------------------
# run() — structure
# ---------------------------------------------------------------------------


def _run_two() -> ExperimentResult:
    dataset = _build_bs_dataset()
    study = DMLStudy(
        inputs=(
            _make_input("H-3-small", "small_price", dataset=dataset, loss="price"),
            _make_input("H-6-small", "small_dml", dataset=dataset, loss="differential"),
        )
    )
    return study.run()


def _run_three() -> ExperimentResult:
    dataset = _build_bs_dataset()
    study = DMLStudy(
        inputs=(
            _make_input("H-3-small", "small_price", dataset=dataset, loss="price"),
            _make_input("H-6-small", "small_dml", dataset=dataset, loss="differential"),
            _make_input("H-3", "baseline_large", dataset=dataset, loss="price"),
        )
    )
    return study.run()


def test_run_returns_experiment_result_with_expected_metadata() -> None:
    result = _run_two()
    assert isinstance(result, ExperimentResult)
    assert result.experiment_id == "E5"
    assert set(result.surrogates) == {"H-3-small", "H-6-small"}
    assert "MAE_Delta" in result.metric_primary


def test_run_emits_one_row_per_bin_per_surrogate_two_inputs() -> None:
    result = _run_two()
    assert len(result.table) == 50  # 2 surrogates x 25 bins


def test_run_emits_one_row_per_bin_per_surrogate_three_inputs() -> None:
    result = _run_three()
    assert len(result.table) == 75  # 3 surrogates x 25 bins


def test_table_carries_expected_columns() -> None:
    result = _run_two()
    expected = {
        "surrogate_id",
        "role",
        "loss",
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


def test_role_and_loss_columns_populated_from_labels() -> None:
    result = _run_two()
    roles_by_surrogate = {
        row["surrogate_id"]: row["role"] for row in result.table
    }
    loss_by_surrogate = {
        row["surrogate_id"]: row["loss"] for row in result.table
    }
    assert roles_by_surrogate["H-3-small"] == "small_price"
    assert roles_by_surrogate["H-6-small"] == "small_dml"
    assert loss_by_surrogate["H-3-small"] == "price"
    assert loss_by_surrogate["H-6-small"] == "differential"


def test_reports_carry_delta_aggregates() -> None:
    result = _run_two()
    for report in result.reports.values():
        assert report.delta is not None


def test_reports_have_no_iv_aggregates() -> None:
    # E5 must not invert IV — it's neither primary nor secondary metric.
    result = _run_two()
    for report in result.reports.values():
        assert report.iv is None


def test_run_raises_when_dataset_has_no_deltas() -> None:
    dataset_with_deltas = _build_bs_dataset()
    dataset_no_delta = OptionDataset(
        features=dataset_with_deltas.features,
        prices=dataset_with_deltas.prices,
        deltas=None,
        raw_inputs=dataset_with_deltas.raw_inputs,
        input_names=dataset_with_deltas.input_names,
    )
    study = DMLStudy(
        inputs=(
            _make_input("H-3-small", "small_price", dataset=dataset_no_delta),
            _make_input("H-6-small", "small_dml", dataset=dataset_no_delta),
        )
    )
    with pytest.raises(RuntimeError, match="Delta"):
        study.run()


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def test_decide_verdict_strong() -> None:
    assert decide_verdict(0.25, 0.05) == "positivo_fuerte"


def test_decide_verdict_strong_with_price_improvement() -> None:
    # Delta mejora >=20% y precio MEJORA (degradation negativa).
    assert decide_verdict(0.25, -0.05) == "positivo_fuerte"


def test_decide_verdict_weak_small_delta_gain() -> None:
    assert decide_verdict(0.10, 0.05) == "positivo_debil"


def test_decide_verdict_negative_no_delta_improvement() -> None:
    assert decide_verdict(0.0, 0.0) == "negativo"
    assert decide_verdict(-0.05, 0.0) == "negativo"


def test_decide_verdict_negative_too_much_price_degradation() -> None:
    # Delta improves a lot but price degrades > 10%
    assert decide_verdict(0.30, 0.15) == "negativo"


def test_run_populates_verdict_field() -> None:
    result = _run_two()
    assert result.verdict in {"positivo_fuerte", "positivo_debil", "negativo"}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_mentions_both_surrogates_and_verdict() -> None:
    result = _run_two()
    summary = result.summary
    assert "H-3-small" in summary
    assert "H-6-small" in summary
    assert "Veredicto" in summary


def test_summary_includes_baseline_distance_when_three_inputs() -> None:
    result = _run_three()
    summary = result.summary
    assert "H-3" in summary
    assert "Distancia" in summary or "distancia" in summary.lower()


def test_summary_omits_baseline_distance_when_two_inputs() -> None:
    result = _run_two()
    assert "Distancia" not in result.summary
    assert "distancia" not in result.summary


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_to_csv_writes_header_plus_rows(tmp_path: Path) -> None:
    result = _run_two()
    output = tmp_path / "e5.csv"
    result.to_csv(output)

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert len(rows) == 50
    roles = {row["role"] for row in rows}
    assert roles == {"small_price", "small_dml"}


def test_to_heatmaps_writes_price_and_delta_per_surrogate(tmp_path: Path) -> None:
    result = _run_two()
    written = result.to_heatmaps(tmp_path, metrics=("price", "delta"))
    # 2 surrogates x 2 metrics = 4 PNGs
    assert len(written) == 4
    for path in written:
        assert path.exists()
        assert path.suffix == ".png"
