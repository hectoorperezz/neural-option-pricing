"""Base común para los experimentos definidos en ``docs/tasks.md``.

Según ``docs/architecture.md``:

    "Cada uno de los cinco experimentos del proyecto se materializa como una
    clase concreta que hereda de `Experiment`. La clase base define el
    método `run`, que es lo único que los scripts de análisis llaman."

Este módulo define:

* ``SurrogateInput``: empaqueta un surrogate, su dataset de test y el
  evaluador que debe usar.
* ``ExperimentResult``: salida uniforme de los experimentos, con tabla,
  resumen y reports internos para generar heatmaps.
* ``Experiment``: clase base abstracta con un único método obligatorio,
  ``run``.

E4 vive fuera de esta jerarquía porque mide tiempos y no usa
``BinEvaluator``.
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
    """Paquete mínimo para evaluar un surrogate dentro de un experimento.

    Permite mezclar surrogates de distintas familias en un mismo estudio. El
    ``evaluator`` ya trae el solver y la partición que corresponden.

    ``labels`` guarda metadatos categóricos como ``activation`` (E2),
    ``sampler`` (E3), ``loss`` o ``role`` (E5). Si falta una clave, el
    experimento debe tratarla como cadena vacía.
    """

    surrogate_id: str
    model: nn.Module
    dataset: OptionDataset
    evaluator: BinEvaluator
    bin_id: np.ndarray | None = None
    labels: dict[str, str] | None = None


@dataclass(frozen=True)
class ExperimentResult:
    """Salida uniforme de ``Experiment.run``.

    ``table`` es la tabla larga por ``(surrogate, bin)`` que se escribe a
    CSV. ``reports`` conserva los ``Report`` originales para poder generar
    heatmaps sin recalcular métricas.

    ``verdict`` solo se rellena cuando la metodología define umbrales
    pre-registrados, como en E3 y E5. En E1 y E2 permanece en ``None``.
    """

    experiment_id: str
    surrogates: tuple[str, ...]
    metric_primary: str
    table: tuple[dict[str, Any], ...]
    summary: str
    reports: dict[str, Report]
    verdict: str | None = None

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
        """Escribe la tabla larga a CSV."""
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
        """Guarda un heatmap PNG por ``(surrogate, metric)``.

        Devuelve las rutas escritas. Por defecto genera precio e IV, que son
        los dos heatmaps requeridos en E1.
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
    """Base abstracta de los experimentos definidos en ``tasks.md``."""

    @abstractmethod
    def run(self) -> ExperimentResult:
        """Ejecuta el experimento y devuelve su resultado."""
