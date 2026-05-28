"""E4 — Eficiencia computacional.

Composes :class:`src.evaluation.timing.TimingBenchmark` for the surrogate
designated by ``docs/tasks.md`` §E4 ("H-3") and produces an
:class:`EfficiencyResult` that mirrors the shape of
:class:`src.experiments.base.ExperimentResult` (table, summary,
``to_csv``) but is **not** an ``Experiment`` subclass.

The reason ``EfficiencyStudy`` lives outside the ``Experiment``
hierarchy is documented in ``src/experiments/base.py``:

    "E4 (the timing benchmark) intentionally lives outside this
    hierarchy because it consumes a different protocol (no
    `BinEvaluator`)."

The methodology document also forbids any fuerte/débil/negativo verdict
for E4 ("es de medición, no de comparación"), so the result has no
``verdict`` field.

The rendered table contains one row per ``(device, batch_size)`` pair
with the median, p25, p75 of solver and surrogate times, the median
speedup and the surrogate throughput in options per second. A line plot
``speedup`` vs ``batch_size`` is produced as the figure deliverable.
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
    """Uniform output of :meth:`EfficiencyStudy.run`."""

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
        """Write the per-(device, batch_size) table as CSV."""
        import csv

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self.table[0].keys())
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.table:
                writer.writerow(row)

    def to_plot(self, path: str | Path) -> Path:
        """Save the speedup-vs-batch line plot as PNG.

        One line per device, log scale on both axes (the speedup spans
        orders of magnitude with the batch size, and the batch sizes
        themselves are powers of ten by construction).
        """
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for device, device_results in self.timings.items():
            batch_sizes = [r.batch_size for r in device_results]
            speedups = [r.speedup_median for r in device_results]
            ax.plot(batch_sizes, speedups, marker="o", label=device)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Tamaño de lote")
        ax.set_ylabel("Speedup (solver / surrogate)")
        ax.set_title(f"E4 — Speedup por tamaño de lote ({self.surrogate_id})")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path


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
