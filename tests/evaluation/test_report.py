import csv
from pathlib import Path

import numpy as np
import pytest

from src.evaluation import BinPartition, Report


def _full_aggregate(n_bins: int, fill: float = 0.0) -> dict[str, np.ndarray]:
    """Build an aggregate dict in the same shape as ``aggregate_by_bin``."""
    return {
        "mean": np.full(n_bins, fill, dtype=np.float64),
        "count": np.full(n_bins, 100, dtype=np.int64),
        "p50": np.full(n_bins, fill, dtype=np.float64),
        "p95": np.full(n_bins, fill, dtype=np.float64),
        "p99": np.full(n_bins, fill, dtype=np.float64),
    }


def _build_full_report() -> Report:
    partition = BinPartition.default()
    n = partition.n_bins
    return Report(
        surrogate_id="BS-3",
        test_path="data/bs_test_6250k_balanced_delta.npz",
        n_samples=6_250_000,
        partition=partition,
        price=_full_aggregate(n, fill=1e-3),
        delta=_full_aggregate(n, fill=1e-2),
        iv=_full_aggregate(n, fill=5e-3),
        iv_failure_rate_per_bin=np.full(n, 0.02, dtype=np.float64),
    )


def test_report_accepts_all_metrics() -> None:
    report = _build_full_report()

    assert report.surrogate_id == "BS-3"
    assert report.n_samples == 6_250_000
    assert report.partition.n_bins == 25
    assert report.price["mean"][0] == 1e-3


def test_report_accepts_missing_delta_and_iv() -> None:
    partition = BinPartition.default()
    report = Report(
        surrogate_id="BS-3-no-delta",
        test_path="data/foo.npz",
        n_samples=1000,
        partition=partition,
        price=_full_aggregate(partition.n_bins),
        delta=None,
        iv=None,
        iv_failure_rate_per_bin=None,
    )

    assert report.delta is None
    assert report.iv is None
    assert report.iv_failure_rate_per_bin is None


def test_report_rejects_missing_required_keys() -> None:
    partition = BinPartition.default()
    bad_price = {"mean": np.zeros(partition.n_bins)}  # missing count, p50, p95, p99

    with pytest.raises(ValueError, match="missing required keys"):
        Report(
            surrogate_id="x",
            test_path="x",
            n_samples=0,
            partition=partition,
            price=bad_price,
            delta=None,
            iv=None,
            iv_failure_rate_per_bin=None,
        )


def test_report_rejects_wrong_shape_aggregate() -> None:
    partition = BinPartition.default()
    bad = _full_aggregate(partition.n_bins)
    bad["mean"] = np.zeros(partition.n_bins + 1)  # wrong length

    with pytest.raises(ValueError, match="shape"):
        Report(
            surrogate_id="x",
            test_path="x",
            n_samples=0,
            partition=partition,
            price=bad,
            delta=None,
            iv=None,
            iv_failure_rate_per_bin=None,
        )


def test_report_rejects_wrong_shape_failure_rate() -> None:
    partition = BinPartition.default()

    with pytest.raises(ValueError, match="iv_failure_rate_per_bin"):
        Report(
            surrogate_id="x",
            test_path="x",
            n_samples=0,
            partition=partition,
            price=_full_aggregate(partition.n_bins),
            delta=None,
            iv=None,
            iv_failure_rate_per_bin=np.zeros(7),
        )


def test_to_csv_writes_25_rows_with_expected_header(tmp_path: Path) -> None:
    report = _build_full_report()
    output = tmp_path / "report.csv"

    report.to_csv(output)

    rows = list(csv.reader(output.open(encoding="utf-8")))
    assert len(rows) == 26  # 1 header + 25 bins
    header = rows[0]
    assert "bin_id" in header
    assert "price_mae_p95" in header
    assert "delta_mae_p95" in header
    assert "iv_mae_p95" in header
    assert "iv_failure_rate" in header


def test_to_csv_values_match_inputs(tmp_path: Path) -> None:
    report = _build_full_report()
    output = tmp_path / "report.csv"

    report.to_csv(output)

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    first = rows[0]
    assert int(first["bin_id"]) == 0
    assert first["moneyness_label"] == "deep_otm"
    assert first["maturity_label"] == "weekly"
    assert first["bin_label"] == "deep_otm_weekly"
    assert int(first["n_points"]) == 100
    assert float(first["price_mae_mean"]) == pytest.approx(1e-3)
    assert float(first["delta_mae_p95"]) == pytest.approx(1e-2)
    assert float(first["iv_mae_p99"]) == pytest.approx(5e-3)
    assert float(first["iv_failure_rate"]) == pytest.approx(0.02)


def test_to_csv_writes_empty_cells_for_missing_delta(tmp_path: Path) -> None:
    partition = BinPartition.default()
    report = Report(
        surrogate_id="BS",
        test_path="x",
        n_samples=10,
        partition=partition,
        price=_full_aggregate(partition.n_bins),
        delta=None,
        iv=None,
        iv_failure_rate_per_bin=None,
    )
    output = tmp_path / "no_delta.csv"

    report.to_csv(output)

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert rows[0]["delta_mae_mean"] == ""
    assert rows[0]["iv_mae_mean"] == ""
    assert rows[0]["iv_failure_rate"] == ""


def test_to_csv_writes_empty_cell_for_nan(tmp_path: Path) -> None:
    partition = BinPartition.default()
    price = _full_aggregate(partition.n_bins)
    price["p95"][0] = np.nan  # the first bin has no usable data for p95

    report = Report(
        surrogate_id="x",
        test_path="x",
        n_samples=0,
        partition=partition,
        price=price,
        delta=None,
        iv=None,
        iv_failure_rate_per_bin=None,
    )
    output = tmp_path / "nan.csv"

    report.to_csv(output)

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert rows[0]["price_mae_p95"] == ""
    assert rows[0]["price_mae_mean"] != ""  # other cells unaffected


def test_to_csv_accepts_path_as_string(tmp_path: Path) -> None:
    report = _build_full_report()
    output = tmp_path / "as_string.csv"

    report.to_csv(str(output))

    assert output.exists()


def test_to_csv_creates_missing_parent_directory(tmp_path: Path) -> None:
    report = _build_full_report()
    output = tmp_path / "nested" / "more" / "report.csv"

    report.to_csv(output)

    assert output.exists()


def test_to_csv_row_order_matches_bin_id(tmp_path: Path) -> None:
    report = _build_full_report()
    output = tmp_path / "ordered.csv"

    report.to_csv(output)

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    bin_ids = [int(row["bin_id"]) for row in rows]
    assert bin_ids == list(range(25))


def test_csv_round_trip_with_numpy_genfromtxt(tmp_path: Path) -> None:
    report = _build_full_report()
    output = tmp_path / "roundtrip.csv"
    report.to_csv(output)

    data = np.genfromtxt(
        output,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    assert len(data) == 25
    np.testing.assert_allclose(data["price_mae_mean"], 1e-3)


# --- Report.to_heatmap ----------------------------------------------------


def test_to_heatmap_writes_png_for_price(tmp_path: Path) -> None:
    report = _build_full_report()
    output = tmp_path / "price.png"

    report.to_heatmap("price", output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_to_heatmap_writes_png_for_iv(tmp_path: Path) -> None:
    report = _build_full_report()
    output = tmp_path / "iv.png"

    report.to_heatmap("iv", output)

    assert output.exists()


def test_to_heatmap_writes_png_for_iv_failure_rate(tmp_path: Path) -> None:
    report = _build_full_report()
    output = tmp_path / "iv_failure.png"

    report.to_heatmap("iv_failure_rate", output)

    assert output.exists()


def test_to_heatmap_rejects_unknown_metric(tmp_path: Path) -> None:
    report = _build_full_report()
    with pytest.raises(ValueError, match="unknown metric"):
        report.to_heatmap("vega", tmp_path / "x.png")


def test_to_heatmap_rejects_missing_aggregate(tmp_path: Path) -> None:
    partition = BinPartition.default()
    report = Report(
        surrogate_id="BS-3",
        test_path="x",
        n_samples=10,
        partition=partition,
        price=_full_aggregate(partition.n_bins),
        delta=None,
        iv=None,
        iv_failure_rate_per_bin=None,
    )

    with pytest.raises(ValueError, match="not populated"):
        report.to_heatmap("delta", tmp_path / "x.png")

    with pytest.raises(ValueError, match="not populated"):
        report.to_heatmap("iv", tmp_path / "x.png")

    with pytest.raises(ValueError, match="not available"):
        report.to_heatmap("iv_failure_rate", tmp_path / "x.png")


def test_to_heatmap_rejects_unknown_statistic(tmp_path: Path) -> None:
    report = _build_full_report()
    with pytest.raises(ValueError, match="not available"):
        report.to_heatmap("price", tmp_path / "x.png", statistic="p25")


def test_to_heatmap_supports_p95_statistic(tmp_path: Path) -> None:
    report = _build_full_report()
    output = tmp_path / "price_p95.png"

    report.to_heatmap("price", output, statistic="p95")

    assert output.exists()


def test_to_heatmap_renders_nan_cells_without_crashing(tmp_path: Path) -> None:
    partition = BinPartition.default()
    price = _full_aggregate(partition.n_bins, fill=1e-3)
    # Wipe a couple of cells so they show as empty bins
    price["mean"][0] = np.nan
    price["mean"][12] = np.nan

    report = Report(
        surrogate_id="BS-3",
        test_path="x",
        n_samples=10,
        partition=partition,
        price=price,
        delta=None,
        iv=None,
        iv_failure_rate_per_bin=None,
    )
    output = tmp_path / "with_nans.png"

    report.to_heatmap("price", output)

    assert output.exists()
