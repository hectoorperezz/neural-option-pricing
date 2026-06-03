"""E6 — Profundidad de red y learning-rate scheduling.

Implementa el experimento descrito en ``docs/experiments/e6.md``:

    "E6 compara variantes del surrogate Heston de referencia (H-3)
    cambiando la profundidad de la red o introduciendo un scheduler de
    learning rate, manteniendo fijos el dataset, la semilla del
    entrenador, el batch size, la activación y la pérdida. La pregunta
    es si alguna de esas dos vías mejora el surrogate baseline sin
    necesidad de aumentar el volumen de datos."

    "La métrica primaria de E6 será `MAE(C/K)` por bin sobre el test
    balanced. `MAE_Delta` se reporta como diagnóstico complementario
    cuando el test trae Delta de referencia."

    "E6 no aparece pre-registrado en `metodologia.md` y no fija
    umbrales fuerte/débil/negativo. La lectura es observacional: se
    ordenan las variantes por `MAE(C/K)` medio por bin y se contrasta
    con el baseline."

Por eso la clase:

* desactiva la inversión de IV porque E6 no la necesita ni como
  primaria ni como control;
* emite una tabla por bin con precio (primaria) y, cuando el dataset
  lo permite, Delta como diagnóstico;
* genera un ranking observacional ordenado por `MAE(C/K)` medio por
  bin, sin emitir veredicto.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.evaluation.report import Report
from src.experiments.base import Experiment, ExperimentResult, SurrogateInput


@dataclass(frozen=True)
class ArchitectureStudy(Experiment):
    """E6: comparación de profundidad y scheduler por ``MAE(C/K)`` por bin."""

    inputs: tuple[SurrogateInput, ...]

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("ArchitectureStudy needs at least one SurrogateInput")
        roles = [_role_label(item) for item in self.inputs]
        if len(set(roles)) != len(roles):
            raise ValueError(
                f"each role must appear at most once; got {roles}"
            )

    def run(self) -> ExperimentResult:
        rows: list[dict[str, Any]] = []
        reports: dict[str, Report] = {}

        for surrogate_input in self.inputs:
            report = surrogate_input.evaluator.evaluate(
                surrogate=surrogate_input.model,
                dataset=surrogate_input.dataset,
                bin_id=surrogate_input.bin_id,
                compute_iv=False,
                surrogate_id=surrogate_input.surrogate_id,
            )

            role = _role_label(surrogate_input)
            architecture = _architecture_label(surrogate_input)
            rows.extend(
                _build_rows(
                    surrogate_input.surrogate_id, role, architecture, report
                )
            )
            reports[surrogate_input.surrogate_id] = report

        summary = _build_summary(rows)
        return ExperimentResult(
            experiment_id="E6",
            surrogates=tuple(item.surrogate_id for item in self.inputs),
            metric_primary="MAE(C/K) por bin (Delta como diagnóstico)",
            table=tuple(rows),
            summary=summary,
            reports=reports,
        )


def _role_label(surrogate_input: SurrogateInput) -> str:
    if surrogate_input.labels is None:
        return ""
    return str(surrogate_input.labels.get("role", ""))


def _architecture_label(surrogate_input: SurrogateInput) -> str:
    if surrogate_input.labels is None:
        return ""
    return str(surrogate_input.labels.get("architecture", ""))


def _build_rows(
    surrogate_id: str,
    role: str,
    architecture: str,
    report: Report,
) -> list[dict[str, Any]]:
    partition = report.partition
    has_delta = report.delta is not None
    rows: list[dict[str, Any]] = []
    for bin_index in range(partition.n_bins):
        moneyness_idx = bin_index % partition.n_moneyness_bins
        maturity_idx = bin_index // partition.n_moneyness_bins
        row: dict[str, Any] = {
            "surrogate_id": surrogate_id,
            "role": role,
            "architecture": architecture,
            "bin_id": bin_index,
            "moneyness_idx": moneyness_idx,
            "maturity_idx": maturity_idx,
            "bin_label": partition.bin_label(moneyness_idx, maturity_idx),
            "n_points": int(report.price["count"][bin_index]),
            "price_mae_mean": float(report.price["mean"][bin_index]),
            "price_mae_p95": float(report.price["p95"][bin_index]),
            "price_mae_p99": float(report.price["p99"][bin_index]),
        }
        if has_delta:
            row["delta_mae_mean"] = float(report.delta["mean"][bin_index])  # type: ignore[index]
            row["delta_mae_p95"] = float(report.delta["p95"][bin_index])  # type: ignore[index]
            row["delta_mae_p99"] = float(report.delta["p99"][bin_index])  # type: ignore[index]
        else:
            row["delta_mae_mean"] = float("nan")
            row["delta_mae_p95"] = float("nan")
            row["delta_mae_p99"] = float("nan")
        rows.append(row)
    return rows


def _build_summary(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "E6 — No data available."

    by_surrogate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_surrogate.setdefault(row["surrogate_id"], []).append(row)

    stats: list[dict[str, Any]] = []
    for surrogate_id, group in by_surrogate.items():
        role = group[0]["role"]
        architecture = group[0]["architecture"]
        finite_price_means = [
            r["price_mae_mean"] for r in group if np.isfinite(r["price_mae_mean"])
        ]
        finite_price_p95s = [
            r["price_mae_p95"] for r in group if np.isfinite(r["price_mae_p95"])
        ]
        finite_delta_means = [
            r["delta_mae_mean"] for r in group if np.isfinite(r["delta_mae_mean"])
        ]
        stats.append(
            {
                "surrogate_id": surrogate_id,
                "role": role,
                "architecture": architecture,
                "price_mae_mean_avg": (
                    float(np.mean(finite_price_means))
                    if finite_price_means
                    else float("nan")
                ),
                "price_mae_p95_worst": (
                    float(max(finite_price_p95s))
                    if finite_price_p95s
                    else float("nan")
                ),
                "delta_mae_mean_avg": (
                    float(np.mean(finite_delta_means))
                    if finite_delta_means
                    else float("nan")
                ),
            }
        )

    stats.sort(
        key=lambda s: s["price_mae_mean_avg"]
        if np.isfinite(s["price_mae_mean_avg"])
        else float("inf")
    )

    lines: list[str] = [
        "E6 — Profundidad y learning-rate scheduling (observacional; sin clasificación fuerte/débil).",
        "Ranking por MAE(C/K) medio por bin (ascendente = mejor):",
    ]
    for stat in stats:
        role = stat["role"] or "(sin rol)"
        architecture = stat["architecture"] or "(sin arch)"
        delta_field = (
            f"{stat['delta_mae_mean_avg']:.4e}"
            if np.isfinite(stat["delta_mae_mean_avg"])
            else "       n/a"
        )
        lines.append(
            f"  {stat['surrogate_id']:>16}  role={role:<12} arch={architecture:<14} "
            f"price_mae_mean={stat['price_mae_mean_avg']:.4e}  "
            f"price_p95_worst_bin={stat['price_mae_p95_worst']:.4e}  "
            f"delta_mae_mean={delta_field}"
        )

    baseline = next(
        (s for s in stats if s["role"] == "baseline"),
        None,
    )
    if baseline is not None and len(stats) > 1:
        lines.append("Distancia frente al baseline (positivo = mejor que baseline):")
        baseline_price = baseline["price_mae_mean_avg"]
        for stat in stats:
            if stat["surrogate_id"] == baseline["surrogate_id"]:
                continue
            improvement = _relative_change(baseline_price, stat["price_mae_mean_avg"])
            improvement_str = (
                f"{improvement:+.2%}"
                if np.isfinite(improvement)
                else "n/a"
            )
            lines.append(
                f"  {stat['surrogate_id']:>16}  vs {baseline['surrogate_id']}: "
                f"{improvement_str}"
            )

    return "\n".join(lines)


def _relative_change(baseline: float, candidate: float) -> float:
    """``(baseline - candidate) / baseline``. Positivo = candidate mejor."""
    if (
        not np.isfinite(baseline)
        or baseline == 0.0
        or not np.isfinite(candidate)
    ):
        return float("nan")
    return (baseline - candidate) / baseline
