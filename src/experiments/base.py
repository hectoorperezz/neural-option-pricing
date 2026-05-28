"""Abstract base for the five experiments planned in ``docs/tasks.md``.

Per ``docs/architecture.md`` §Experiments:

    "Cada uno de los cinco experimentos del proyecto se materializa como una
    clase concreta que hereda de `Experiment`. La clase base define el
    método `run`, que es lo único que los scripts de análisis llaman."

This module exposes:

* :class:`SurrogateInput` — a frozen dataclass bundling one surrogate
  with the test dataset and the :class:`BinEvaluator` that should
  evaluate it. Each experiment that walks several surrogates takes a
  tuple of these.
* :class:`ExperimentResult` — the uniform return type of every
  ``Experiment.run()``. Holds the per-bin table, the observational
  summary, the experiment identifier and the underlying
  :class:`Report` objects so consumers can also derive heatmaps.
* :class:`Experiment` — the abstract base with the single required
  method ``run`` returning an :class:`ExperimentResult`.

E4 (the timing benchmark) intentionally lives outside this hierarchy
because it consumes a different protocol (no :class:`BinEvaluator`).
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch import nn

from src.datasets.generator import OptionDataset
from src.evaluation.binevaluator import BinEvaluator
from src.evaluation.report import Report


@dataclass(frozen=True)
class SurrogateInput:
    """One bundle ready to be evaluated by an :class:`Experiment`.

    Each experiment that consumes :class:`BinEvaluator` receives a tuple
    of these so that surrogates of different families (BS vs Heston) can
    coexist in the same study. The ``evaluator`` already carries the
    matching solver and partition.

    ``labels`` is an optional categorical metadata dictionary that
    experiments such as :class:`ActivationStudy` (E2),
    ``SamplingStudy`` (E3) or ``DMLStudy`` (E5) read to fill the table
    column that distinguishes the dimension being varied. Keys used so
    far: ``"activation"`` (E2), ``"sampler"`` (E3), ``"loss"`` (E5).
    Experiments must tolerate a missing key gracefully (empty string).
    """

    surrogate_id: str
    model: nn.Module
    dataset: OptionDataset
    evaluator: BinEvaluator
    bin_id: np.ndarray | None = None
    labels: dict[str, str] | None = None


@dataclass(frozen=True)
class ExperimentResult:
    """Uniform output of every :class:`Experiment.run` invocation.

    ``table`` is the long-format per-(surrogate, bin) table that the CSV
    serializer writes verbatim. ``reports`` keeps the underlying
    :class:`Report` instances so callers can additionally produce
    heatmaps via :meth:`Report.to_heatmap`.
    """

    experiment_id: str
    surrogates: tuple[str, ...]
    metric_primary: str
    table: tuple[dict[str, Any], ...]
    summary: str
    reports: dict[str, Report]

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id must be a non-empty string")
        if not self.surrogates:
            raise ValueError("at least one surrogate must be present in the result")
        missing = set(self.surrogates) - set(self.reports)
        if missing:
            raise ValueError(
                f"reports dict is missing entries for surrogates: {sorted(missing)}"
            )

    def to_csv(self, path: str | Path) -> None:
        """Write the long-format table to ``path``."""
        if not self.table:
            raise ValueError("cannot serialize an empty table")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self.table[0].keys())
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.table:
                writer.writerow(self._render_row(row))

    def to_heatmaps(
        self,
        directory: str | Path,
        metrics: tuple[str, ...] = ("price", "iv"),
        statistic: str = "mean",
    ) -> list[Path]:
        """Save one heatmap PNG per (surrogate, metric) into ``directory``.

        Returns the list of paths written. By default produces the two
        figures that ``tasks.md`` §E1 requires ("heatmaps separados para
        precio e IV"); other experiments can request additional metrics.
        """
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for surrogate_id, report in self.reports.items():
            for metric in metrics:
                output_path = output_dir / f"{surrogate_id}_{metric}_{statistic}.png"
                report.to_heatmap(metric, output_path, statistic=statistic)
                written.append(output_path)
        return written

    @staticmethod
    def _render_row(row: dict[str, Any]) -> dict[str, Any]:
        rendered: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, float) and not np.isfinite(value):
                rendered[key] = ""
            else:
                rendered[key] = value
        return rendered


class Experiment(ABC):
    """Abstract base for the five experiments documented in ``tasks.md``."""

    @abstractmethod
    def run(self) -> ExperimentResult:
        """Execute the experiment and return its :class:`ExperimentResult`."""
