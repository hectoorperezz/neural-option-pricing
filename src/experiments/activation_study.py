"""E2 — Activaciones y calidad de Delta.

Implementa el experimento descrito en ``docs/tasks.md`` y
``docs/metodologia.md``:

    "E2 compara ReLU, Softplus, Swish y tanh manteniendo fijo todo lo
    demás. La pregunta no es solo qué activación da mejor precio, sino
    cuál produce mejores derivadas."

    "La métrica primaria de E2 será `MAE_Delta` por bin. El precio
    normalizado se mantiene como control, porque una activación solo
    será preferible si mejora las derivadas sin degradar de forma
    relevante el nivel de la función."

    "No fijamos umbrales fuerte/débil para E2."

Por eso la clase:

* desactiva IV porque E2 evalúa Delta y precio como control;
* exige que el test set traiga Delta de referencia;
* emite una tabla por bin con precio, Delta y activación;
* genera un ranking observacional, sin veredicto fuerte/débil.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.evaluation.report import Report
from src.experiments.base import Experiment, ExperimentResult, SurrogateInput


@dataclass(frozen=True)
class ActivationStudy(Experiment):
    """E2: comparación de activaciones por ``MAE_Delta`` por bin."""

    inputs: tuple[SurrogateInput, ...]
    family_label: str = ""

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("ActivationStudy needs at least one SurrogateInput")

    def run(self) -> ExperimentResult:
        rows: list[dict[str, Any]] = []
        reports: dict[str, Report] = {}

        for surrogate_input in self.inputs:
            if surrogate_input.dataset.deltas is None:
                raise RuntimeError(
                    f"surrogate {surrogate_input.surrogate_id} has no Delta column "
                    "in its test dataset; E2's primary metric is MAE_Delta and "
                    "the comparison cannot proceed without it"
                )

            report = surrogate_input.evaluator.evaluate(
                surrogate=surrogate_input.model,
                dataset=surrogate_input.dataset,
                bin_id=surrogate_input.bin_id,
                compute_iv=False,
                surrogate_id=surrogate_input.surrogate_id,
            )
            if report.delta is None:
                raise RuntimeError(
                    f"BinEvaluator returned no Delta aggregates for "
                    f"{surrogate_input.surrogate_id}; check the dataset"
                )

            activation = _activation_label(surrogate_input)
            rows.extend(_build_rows(surrogate_input.surrogate_id, activation, report))
            reports[surrogate_input.surrogate_id] = report

        summary = _build_summary(rows, self.family_label)
        return ExperimentResult(
            experiment_id="E2",
            surrogates=tuple(item.surrogate_id for item in self.inputs),
            metric_primary="MAE_Delta por bin (precio como control)",
            table=tuple(rows),
            summary=summary,
            reports=reports,
        )


def _activation_label(surrogate_input: SurrogateInput) -> str:
    if surrogate_input.labels is None:
        return ""
    return str(surrogate_input.labels.get("activation", ""))


def _build_rows(
    surrogate_id: str,
    activation: str,
    report: Report,
) -> list[dict[str, Any]]:
    partition = report.partition
    rows: list[dict[str, Any]] = []
    for bin_index in range(partition.n_bins):
        moneyness_idx = bin_index % partition.n_moneyness_bins
        maturity_idx = bin_index // partition.n_moneyness_bins
        rows.append(
            {
                "surrogate_id": surrogate_id,
                "activation": activation,
                "bin_id": bin_index,
                "moneyness_idx": moneyness_idx,
                "maturity_idx": maturity_idx,
                "bin_label": partition.bin_label(moneyness_idx, maturity_idx),
                "n_points": int(report.price["count"][bin_index]),
                "price_mae_mean": float(report.price["mean"][bin_index]),
                "price_mae_p95": float(report.price["p95"][bin_index]),
                "price_mae_p99": float(report.price["p99"][bin_index]),
                "delta_mae_mean": float(report.delta["mean"][bin_index]),  # type: ignore[index]
                "delta_mae_p95": float(report.delta["p95"][bin_index]),  # type: ignore[index]
                "delta_mae_p99": float(report.delta["p99"][bin_index]),  # type: ignore[index]
            }
        )
    return rows


def _build_summary(rows: list[dict[str, Any]], family_label: str) -> str:
    if not rows:
        return "E2 — No data available."

    by_activation: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_activation.setdefault(row["activation"], []).append(row)

    stats: list[dict[str, Any]] = []
    for activation, group in by_activation.items():
        finite_delta_means = [
            r["delta_mae_mean"] for r in group if np.isfinite(r["delta_mae_mean"])
        ]
        finite_delta_p95s = [
            r["delta_mae_p95"] for r in group if np.isfinite(r["delta_mae_p95"])
        ]
        finite_price_means = [
            r["price_mae_mean"] for r in group if np.isfinite(r["price_mae_mean"])
        ]
        stats.append(
            {
                "activation": activation,
                "delta_mae_mean_avg": (
                    float(np.mean(finite_delta_means)) if finite_delta_means else float("nan")
                ),
                "delta_mae_p95_worst": (
                    float(max(finite_delta_p95s)) if finite_delta_p95s else float("nan")
                ),
                "price_mae_mean_avg": (
                    float(np.mean(finite_price_means)) if finite_price_means else float("nan")
                ),
            }
        )

    stats.sort(
        key=lambda s: s["delta_mae_mean_avg"]
        if np.isfinite(s["delta_mae_mean_avg"])
        else float("inf")
    )

    header = (
        f"E2 — Comparación de activaciones en {family_label}"
        if family_label
        else "E2 — Comparación de activaciones"
    )
    lines: list[str] = [
        f"{header} (observacional; sin clasificación fuerte/débil).",
        "Ranking por MAE_Delta medio por bin (ascendente = mejor):",
    ]
    for stat in stats:
        label = stat["activation"] or "(sin etiqueta)"
        lines.append(
            f"  {label:>12}  delta_mae_mean={stat['delta_mae_mean_avg']:.4e}  "
            f"delta_p95_worst_bin={stat['delta_mae_p95_worst']:.4e}  "
            f"price_mae_mean={stat['price_mae_mean_avg']:.4e}"
        )
    return "\n".join(lines)
