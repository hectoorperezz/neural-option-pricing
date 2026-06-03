"""E4 — Eficiencia computacional.

Compone ``TimingBenchmark`` para el surrogate fijado en E4 (H-3) y devuelve
un ``EfficiencyResult`` con tabla, resumen y serialización a CSV. No hereda
de ``Experiment`` porque su protocolo no usa ``BinEvaluator``.

La metodología no define veredicto fuerte/débil/negativo para E4: es un
experimento de medición, no de comparación.

La tabla contiene una fila por ``(device, batch_size)`` con mediana, p25,
p75, speedup y throughput del surrogate. La figura asociada representa
``speedup`` frente a ``batch_size``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from torch import nn

from src.evaluation.timing import TimingBenchmark, TimingResult


@dataclass(frozen=True)
class EfficiencyStudy:
    """E4: cronometra H-3 frente al solver Heston por tamaño de lote y device."""

    benchmark: TimingBenchmark
    surrogate: nn.Module
    surrogate_id: str
    devices: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.surrogate_id:
            raise ValueError("surrogate_id must be a non-empty string")
        if not self.devices:
            raise ValueError("devices must contain at least one entry")
        seen: set[str] = set()
        for device in self.devices:
            if device in seen:
                raise ValueError(f"duplicate device entry: {device}")
            seen.add(device)

    def run(
        self, logger: Callable[[str], None] | None = None
    ) -> "EfficiencyResult":
        all_results = self.benchmark.run(
            self.surrogate, self.devices, logger=logger
        )
        timings: dict[str, list[TimingResult]] = {d: [] for d in self.devices}
        for result in all_results:
            timings[result.device].append(result)
        timings_tuple: dict[str, tuple[TimingResult, ...]] = {
            device: tuple(items) for device, items in timings.items()
        }
        rows = [_render_row(self.surrogate_id, r) for r in all_results]

        summary = _build_summary(self.surrogate_id, timings_tuple)
        return EfficiencyResult(
            experiment_id="E4",
            surrogate_id=self.surrogate_id,
            metric_primary="speedup = tiempo_solver / tiempo_surrogate por tamaño de lote",
            table=tuple(rows),
            summary=summary,
            timings=timings_tuple,
        )


@dataclass(frozen=True)
class EfficiencyResult:
    """Salida uniforme de ``EfficiencyStudy.run``."""

    experiment_id: str
    surrogate_id: str
    metric_primary: str
    table: tuple[dict[str, Any], ...]
    summary: str
    timings: dict[str, tuple[TimingResult, ...]]

    def __post_init__(self) -> None:
        if not self.table:
            raise ValueError("table cannot be empty")
        if not self.timings:
            raise ValueError("timings cannot be empty")

    def to_csv(self, path: str | Path) -> None:
        """Escribe la tabla por ``(device, batch_size)`` a CSV."""
        import csv

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self.table[0].keys())
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.table:
                writer.writerow(row)

def _render_row(surrogate_id: str, result: TimingResult) -> dict[str, Any]:
    throughput = (
        float(result.batch_size) / result.surrogate_median_s
        if result.surrogate_median_s > 0.0
        else float("inf")
    )
    return {
        "surrogate_id": surrogate_id,
        "device": result.device,
        "batch_size": result.batch_size,
        "n_repetitions": result.n_repetitions,
        "solver_median_s": result.solver_median_s,
        "solver_p25_s": result.solver_p25_s,
        "solver_p75_s": result.solver_p75_s,
        "surrogate_median_s": result.surrogate_median_s,
        "surrogate_p25_s": result.surrogate_p25_s,
        "surrogate_p75_s": result.surrogate_p75_s,
        "speedup_median": result.speedup_median,
        "throughput_surrogate_options_per_s": throughput,
    }


def _build_summary(
    surrogate_id: str,
    timings: dict[str, tuple[TimingResult, ...]],
) -> str:
    lines = [
        f"E4 — Eficiencia computacional ({surrogate_id} frente al solver).",
    ]
    for device, device_results in timings.items():
        largest = max(device_results, key=lambda r: r.batch_size)
        lines.append(
            f"  {device}: lote más grande = {largest.batch_size}, "
            f"solver mediana = {largest.solver_median_s:.4f}s, "
            f"surrogate mediana = {largest.surrogate_median_s:.4e}s, "
            f"speedup = x{largest.speedup_median:.1f}"
        )
    return "\n".join(lines)
