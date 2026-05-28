import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.evaluation.timing import TimingBenchmark
from src.experiments import EfficiencyResult, EfficiencyStudy
from src.solvers import BlackScholesSolver


class _ConstantPriceModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]


def _build_benchmark(
    *,
    batch_sizes: tuple[int, ...] = (10, 50),
    n_warmups: int = 1,
    n_repetitions: int = 3,
) -> TimingBenchmark:
    rng = np.random.default_rng(11)
    n = 200
    moneyness = rng.uniform(0.6, 1.6, size=n).astype(np.float32)
    maturity = rng.uniform(0.1, 1.0, size=n).astype(np.float32)
    rate = rng.uniform(0.0, 0.05, size=n).astype(np.float32)
    sigma = rng.uniform(0.1, 0.5, size=n).astype(np.float32)
    raw_inputs = np.stack([moneyness, maturity, rate, sigma], axis=1)
    features = raw_inputs.copy()
    return TimingBenchmark(
        pricer=BlackScholesSolver(),
        raw_inputs=raw_inputs,
        features=features,
        input_names=("moneyness", "maturity", "rate", "volatility"),
        batch_sizes=batch_sizes,
        n_warmups=n_warmups,
        n_repetitions=n_repetitions,
    )


def _make_study(
    *,
    devices: tuple[str, ...] = ("cpu",),
    batch_sizes: tuple[int, ...] = (10, 50),
) -> EfficiencyStudy:
    return EfficiencyStudy(
        benchmark=_build_benchmark(batch_sizes=batch_sizes),
        surrogate=_ConstantPriceModel(),
        surrogate_id="H-3",
        devices=devices,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_efficiency_study_rejects_empty_devices() -> None:
    with pytest.raises(ValueError, match="devices"):
        EfficiencyStudy(
            benchmark=_build_benchmark(),
            surrogate=_ConstantPriceModel(),
            surrogate_id="H-3",
            devices=(),
        )


def test_efficiency_study_rejects_duplicate_devices() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        EfficiencyStudy(
            benchmark=_build_benchmark(),
            surrogate=_ConstantPriceModel(),
            surrogate_id="H-3",
            devices=("cpu", "cpu"),
        )


def test_efficiency_study_rejects_empty_surrogate_id() -> None:
    with pytest.raises(ValueError, match="surrogate_id"):
        EfficiencyStudy(
            benchmark=_build_benchmark(),
            surrogate=_ConstantPriceModel(),
            surrogate_id="",
            devices=("cpu",),
        )


# ---------------------------------------------------------------------------
# run() — structure
# ---------------------------------------------------------------------------


def test_run_returns_efficiency_result_with_expected_metadata() -> None:
    result = _make_study().run()
    assert isinstance(result, EfficiencyResult)
    assert result.experiment_id == "E4"
    assert result.surrogate_id == "H-3"
    assert "speedup" in result.metric_primary.lower()


def test_table_has_one_row_per_device_and_batch_size() -> None:
    result = _make_study(batch_sizes=(10, 50)).run()
    assert len(result.table) == 2  # 1 device x 2 batch sizes


def test_table_columns_match_methodology() -> None:
    result = _make_study().run()
    expected = {
        "surrogate_id",
        "device",
        "batch_size",
        "n_repetitions",
        "solver_median_s",
        "solver_p25_s",
        "solver_p75_s",
        "surrogate_median_s",
        "surrogate_p25_s",
        "surrogate_p75_s",
        "speedup_median",
        "throughput_surrogate_options_per_s",
    }
    assert expected.issubset(set(result.table[0].keys()))


def test_summary_mentions_each_device() -> None:
    result = _make_study(devices=("cpu",)).run()
    assert "cpu" in result.summary
    assert "speedup" in result.summary.lower()


def test_timings_dict_indexed_by_device() -> None:
    result = _make_study(devices=("cpu",), batch_sizes=(10, 50, 100)).run()
    assert set(result.timings.keys()) == {"cpu"}
    assert tuple(r.batch_size for r in result.timings["cpu"]) == (10, 50, 100)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_to_csv_writes_header_plus_rows(tmp_path: Path) -> None:
    result = _make_study(batch_sizes=(10, 50)).run()
    output = tmp_path / "e4.csv"
    result.to_csv(output)
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["device"] == "cpu"


def test_to_plot_writes_png(tmp_path: Path) -> None:
    result = _make_study(batch_sizes=(10, 50, 100)).run()
    output = tmp_path / "speedup.png"
    written = result.to_plot(output)
    assert written == output
    assert output.exists()
    assert output.suffix == ".png"
