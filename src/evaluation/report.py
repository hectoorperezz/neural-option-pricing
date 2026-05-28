"""Resultado agregado de evaluar un surrogate por bins.

``Report`` es el contenedor que devuelve ``BinEvaluator``. Guarda métricas ya
agregadas, no recalcula nada, y sabe serializarlas a CSV o a heatmaps.

El CSV usa formato ancho: una fila por bin y una columna por estadístico.
Esto facilita la inspección directa en Excel. Cada métrica reporta media,
p50, p95 y p99, tal como se fijó en ``docs/metodologia.md``. La tasa de fallo
de inversión IV se conserva como columna propia porque forma parte del
diagnóstico, no es un detalle a ocultar.
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
    """Resumen de calidad por bin, serializable a CSV.

    ``price``, ``delta`` e ``iv`` tienen la forma que produce
    ``aggregate_by_bin``: claves ``mean``, ``count``, ``p50``, ``p95`` y
    ``p99``. Cada valor es un array de longitud ``partition.n_bins``.
    ``delta`` e ``iv`` pueden ser ``None`` si el dataset no trae Deltas o si
    se omitió la inversión de IV.

    ``iv_failure_rate_per_bin`` mide la fracción de puntos del bin donde el
    inversor de IV no pudo devolver una volatilidad válida.
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
        """Escribe el resumen por bin en un CSV ancho."""
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

    def to_heatmap(
        self,
        metric: str,
        path: str | Path,
        statistic: str = "mean",
    ) -> None:
        """Guarda un heatmap 5x5 para una métrica y un estadístico.

        ``metric`` puede ser ``"price"``, ``"delta"``, ``"iv"`` o
        ``"iv_failure_rate"``. ``statistic`` puede ser ``"mean"``, ``"p50"``,
        ``"p95"`` o ``"p99"``. En ``iv_failure_rate`` se ignora porque no hay
        percentiles. Los bins vacíos se pintan en gris claro para separarlos
        visualmente del cero numérico.
        """
        values = self._values_for_heatmap(metric, statistic)
        n_rows = self.partition.n_maturity_bins
        n_cols = self.partition.n_moneyness_bins
        grid = np.asarray(values, dtype=np.float64).reshape(n_rows, n_cols)

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        masked = np.ma.masked_invalid(grid)
        figure, axis = plt.subplots(figsize=(8.0, 6.0))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="#dddddd")
        image = axis.imshow(masked, cmap=cmap, aspect="auto", origin="upper")

        axis.set_xticks(range(n_cols))
        axis.set_xticklabels(self.partition.moneyness_labels, rotation=30, ha="right")
        axis.set_yticks(range(n_rows))
        axis.set_yticklabels(self.partition.maturity_labels)
        axis.set_xlabel("moneyness bin")
        axis.set_ylabel("maturity bin")
        title_metric = metric if metric == "iv_failure_rate" else f"{metric} MAE"
        title_statistic = "" if metric == "iv_failure_rate" else f" ({statistic})"
        axis.set_title(f"{self.surrogate_id} — {title_metric}{title_statistic}")

        finite_mean = float(masked.mean()) if masked.count() > 0 else 0.0
        for row in range(n_rows):
            for col in range(n_cols):
                cell = masked[row, col]
                if np.ma.is_masked(cell):
                    continue
                axis.text(
                    col,
                    row,
                    f"{float(cell):.2e}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if float(cell) > finite_mean else "black",
                )

        figure.colorbar(image, ax=axis)
        figure.tight_layout()

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=120)
        plt.close(figure)

    def _values_for_heatmap(self, metric: str, statistic: str) -> np.ndarray:
        if metric == "iv_failure_rate":
            if self.iv_failure_rate_per_bin is None:
                raise ValueError(
                    "iv_failure_rate is not available in this report; "
                    "the evaluator must run with compute_iv=True"
                )
            return np.asarray(self.iv_failure_rate_per_bin)

        aggregate_by_metric: dict[str, dict[str, np.ndarray] | None] = {
            "price": self.price,
            "delta": self.delta,
            "iv": self.iv,
        }
        if metric not in aggregate_by_metric:
            raise ValueError(
                f"unknown metric {metric!r}; expected one of "
                f"{sorted(list(aggregate_by_metric) + ['iv_failure_rate'])}"
            )
        aggregate = aggregate_by_metric[metric]
        if aggregate is None:
            raise ValueError(
                f"metric {metric!r} is not populated in this report "
                "(was the dataset/evaluator configured to compute it?)"
            )
        if statistic not in aggregate:
            raise ValueError(
                f"statistic {statistic!r} not available; "
                f"choose one of {sorted(aggregate)}"
            )
        return np.asarray(aggregate[statistic])
