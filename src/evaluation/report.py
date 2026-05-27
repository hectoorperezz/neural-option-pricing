"""Container for per-bin surrogate evaluation results.

:class:`Report` is the data structure that ``BinEvaluator`` will return after
applying :mod:`src.evaluation.metrics` to a surrogate over a test set. It is
intentionally a thin dataclass that holds already-aggregated values and
knows how to write them out as a CSV; it does no arithmetic itself.

The CSV layout is wide (one row per bin, one column per metric statistic)
because the immediate consumer is a human inspecting results in Excel; the
``Experiment`` classes planned for the next phase can re-aggregate this
into long format with pandas if needed.

Per ``docs/metodologia.md`` ("Las métricas mínimas de evaluación serán
`MAE(C/K)`, `MAE_IV` y `MAE_Delta`. Para cada una se reportarán promedios y
percentiles por bin, con especial atención al percentil 95.") every metric
shows mean and p50/p95/p99 in the CSV. The IV failure rate per bin lives in
its own column because the methodology document explicitly mandates that it
be surfaced as a diagnostic ("los fallos no deben ocultarse, se reportarán
como parte del diagnóstico").
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.evaluation.binning import BinPartition


_METRICS_IN_CSV: tuple[str, ...] = ("mean", "p50", "p95", "p99")


@dataclass(frozen=True)
class Report:
    """Per-bin surrogate quality summary, with CSV serialization.

    ``price``, ``delta`` and ``iv`` are dicts in the shape produced by
    :func:`src.evaluation.metrics.aggregate_by_bin`: keys ``mean``,
    ``count``, ``p50``, ``p95``, ``p99``; each value is an ``np.ndarray``
    of length ``partition.n_bins``. ``delta`` and ``iv`` may be ``None``
    if the dataset did not provide Deltas or if IV inversion was skipped.

    ``iv_failure_rate_per_bin`` is the fraction of points in each bin for
    which the BS implied-volatility inverter could not return a value;
    methodology requires that it be reported alongside ``MAE_IV``.
    """

    surrogate_id: str
    test_path: str
    n_samples: int
    partition: BinPartition
    price: dict[str, np.ndarray]
    delta: dict[str, np.ndarray] | None
    iv: dict[str, np.ndarray] | None
    iv_failure_rate_per_bin: np.ndarray | None

    def __post_init__(self) -> None:
        self._validate_aggregate(self.price, "price")
        if self.delta is not None:
            self._validate_aggregate(self.delta, "delta")
        if self.iv is not None:
            self._validate_aggregate(self.iv, "iv")
        if self.iv_failure_rate_per_bin is not None:
            failure = np.asarray(self.iv_failure_rate_per_bin)
            if failure.shape != (self.partition.n_bins,):
                raise ValueError(
                    "iv_failure_rate_per_bin must have shape "
                    f"({self.partition.n_bins},), got {failure.shape}"
                )

    def _validate_aggregate(self, agg: dict[str, np.ndarray], name: str) -> None:
        required = {"mean", "count", "p50", "p95", "p99"}
        missing = required - set(agg)
        if missing:
            raise ValueError(
                f"{name} aggregate is missing required keys: {sorted(missing)}"
            )
        for key in required:
            array = np.asarray(agg[key])
            if array.shape != (self.partition.n_bins,):
                raise ValueError(
                    f"{name}[{key!r}] must have shape ({self.partition.n_bins},), "
                    f"got {array.shape}"
                )

    def to_csv(self, path: str | Path) -> None:
        """Write the per-bin summary to a wide CSV file at ``path``."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(self._csv_header())
            for bin_index in range(self.partition.n_bins):
                writer.writerow(self._csv_row(bin_index))

    def _csv_header(self) -> list[str]:
        header: list[str] = [
            "bin_id",
            "moneyness_idx",
            "maturity_idx",
            "moneyness_label",
            "maturity_label",
            "bin_label",
            "n_points",
        ]
        for metric in _METRICS_IN_CSV:
            header.append(f"price_mae_{metric}")
        for metric in _METRICS_IN_CSV:
            header.append(f"delta_mae_{metric}")
        for metric in _METRICS_IN_CSV:
            header.append(f"iv_mae_{metric}")
        header.append("iv_failure_rate")
        return header

    def _csv_row(self, bin_index: int) -> list[object]:
        moneyness_idx = bin_index % self.partition.n_moneyness_bins
        maturity_idx = bin_index // self.partition.n_moneyness_bins
        moneyness_label = self.partition.moneyness_labels[moneyness_idx]
        maturity_label = self.partition.maturity_labels[maturity_idx]
        bin_label = self.partition.bin_label(moneyness_idx, maturity_idx)
        n_points = int(self.price["count"][bin_index])

        row: list[object] = [
            bin_index,
            moneyness_idx,
            maturity_idx,
            moneyness_label,
            maturity_label,
            bin_label,
            n_points,
        ]
        row.extend(self._aggregate_cells(self.price, bin_index))
        row.extend(self._aggregate_cells(self.delta, bin_index))
        row.extend(self._aggregate_cells(self.iv, bin_index))
        row.append(self._format_cell(self.iv_failure_rate_per_bin, bin_index))
        return row

    @staticmethod
    def _aggregate_cells(
        aggregate: dict[str, np.ndarray] | None,
        bin_index: int,
    ) -> list[object]:
        if aggregate is None:
            return ["" for _ in _METRICS_IN_CSV]
        return [Report._format_cell(aggregate[name], bin_index) for name in _METRICS_IN_CSV]

    @staticmethod
    def _format_cell(array: np.ndarray | None, bin_index: int) -> object:
        if array is None:
            return ""
        value = float(np.asarray(array)[bin_index])
        if not np.isfinite(value):
            return ""
        return value
