"""E3 — Muestreo uniforme frente a enfocado.

Implementa el experimento descrito en ``docs/tasks.md`` y
``docs/metodologia.md``:

    "La comparación principal será H-3 frente a H-5. Ambos surrogates
    deben compartir arquitectura, pérdida, optimizador, tamaño de
    dataset y protocolo de evaluación."

    "La métrica primaria de E3 será `MAE_IV` en los bins ATM combinados
    con weekly, short y medium-short."

    "Fijamos antes de entrenar tres niveles de interpretación. E3 será
    positivo fuerte si H-5 reduce al menos un 10% el `MAE_IV` promedio
    en los bins críticos frente a H-3 y el `MAE_IV` global no empeora
    más de un 10%. Será positivo débil si H-5 mejora esos bins pero
    menos de un 10%, o si la mejora local viene acompañada de un
    deterioro global moderado. Será negativo si no mejora los bins
    críticos o si la mejora local exige sacrificar de forma excesiva
    la cobertura global."

Por eso la clase:

* recibe exactamente dos surrogates: ``uniform`` y ``focused``;
* fuerza ``compute_iv=True`` porque IV es la métrica primaria;
* agrega ``MAE_IV`` en ``atm_weekly``, ``atm_short`` y
  ``atm_medium_short``;
* compara esa mejora local con el deterioro global y asigna el veredicto.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.evaluation.report import Report
from src.experiments.base import Experiment, ExperimentResult, SurrogateInput


# Bins críticos definidos en tasks.md §E3 / metodologia.md §E3:
# "ATM combinado con weekly, short y medium-short".
_CRITICAL_BIN_LABELS: tuple[str, ...] = (
    "atm_weekly",
    "atm_short",
    "atm_medium_short",
)

# Umbrales tomados de metodologia.md §E3.
_STRONG_IMPROVEMENT_THRESHOLD: float = 0.10
_STRONG_MAX_GLOBAL_DEGRADATION: float = 0.10

# La metodología distingue deterioro global moderado y excesivo sin fijar un
# número. Usamos 20% como frontera operativa: por debajo sigue siendo
# positivo débil; por encima pasa a negativo.
_EXCESSIVE_GLOBAL_DEGRADATION: float = 0.20


@dataclass(frozen=True)
class SamplingStudy(Experiment):
    """E3: ``MAE_IV`` en bins críticos, uniforme (H-3) vs enfocado (H-5)."""

    inputs: tuple[SurrogateInput, ...]

    def __post_init__(self) -> None:
        if len(self.inputs) != 2:
            raise ValueError(
                "SamplingStudy compares exactly two surrogates "
                "(one uniform, one focused); got "
                f"{len(self.inputs)}"
            )
        samplers = [_sampler_label(item) for item in self.inputs]
        if "uniform" not in samplers or "focused" not in samplers:
            raise ValueError(
                "SamplingStudy expects one input with labels['sampler']='uniform' "
                "and one with labels['sampler']='focused'; got "
                f"{samplers}"
            )
        if samplers[0] == samplers[1]:
            raise ValueError("the two inputs must carry distinct samplers")

    def run(self) -> ExperimentResult:
        reports: dict[str, Report] = {}
        rows: list[dict[str, Any]] = []

        uniform_id = ""
        focused_id = ""
        for surrogate_input in self.inputs:
            sampler = _sampler_label(surrogate_input)
            if sampler == "uniform":
                uniform_id = surrogate_input.surrogate_id
            else:
                focused_id = surrogate_input.surrogate_id

            report = surrogate_input.evaluator.evaluate(
                surrogate=surrogate_input.model,
                dataset=surrogate_input.dataset,
                bin_id=surrogate_input.bin_id,
                compute_iv=True,
                surrogate_id=surrogate_input.surrogate_id,
            )
            if report.iv is None:
                raise RuntimeError(
                    f"BinEvaluator returned no IV aggregates for "
                    f"{surrogate_input.surrogate_id}; E3 cannot proceed"
                )
            reports[surrogate_input.surrogate_id] = report
            rows.extend(_build_rows(surrogate_input.surrogate_id, sampler, report))

        uniform_critical = _mean_mae_iv_over_critical_bins(reports[uniform_id])
        focused_critical = _mean_mae_iv_over_critical_bins(reports[focused_id])
        uniform_global = _global_mae_iv(reports[uniform_id])
        focused_global = _global_mae_iv(reports[focused_id])

        improvement_critical = _relative_change(uniform_critical, focused_critical)
        global_degradation = _relative_degradation(uniform_global, focused_global)
        verdict = decide_verdict(improvement_critical, global_degradation)

        summary = _build_summary(
            uniform_id=uniform_id,
            focused_id=focused_id,
            uniform_critical=uniform_critical,
            focused_critical=focused_critical,
            uniform_global=uniform_global,
            focused_global=focused_global,
            improvement_critical=improvement_critical,
            global_degradation=global_degradation,
            verdict=verdict,
        )

        return ExperimentResult(
            experiment_id="E3",
            surrogates=tuple(item.surrogate_id for item in self.inputs),
            metric_primary="MAE_IV en bins ATM × {weekly, short, medium_short}",
            table=tuple(rows),
            summary=summary,
            reports=reports,
            verdict=verdict,
        )


def decide_verdict(improvement_critical: float, global_degradation: float) -> str:
    """Aplica los umbrales documentados en ``metodologia.md`` §E3."""
    if not np.isfinite(improvement_critical) or improvement_critical <= 0.0:
        return "negativo"
    if np.isfinite(global_degradation) and global_degradation > _EXCESSIVE_GLOBAL_DEGRADATION:
        return "negativo"
    if (
        improvement_critical >= _STRONG_IMPROVEMENT_THRESHOLD
        and (not np.isfinite(global_degradation) or global_degradation <= _STRONG_MAX_GLOBAL_DEGRADATION)
    ):
        return "positivo_fuerte"
    return "positivo_debil"


def _sampler_label(surrogate_input: SurrogateInput) -> str:
    if surrogate_input.labels is None:
        return ""
    return str(surrogate_input.labels.get("sampler", ""))


def _build_rows(surrogate_id: str, sampler: str, report: Report) -> list[dict[str, Any]]:
    partition = report.partition
    rows: list[dict[str, Any]] = []
    for bin_index in range(partition.n_bins):
        moneyness_idx = bin_index % partition.n_moneyness_bins
        maturity_idx = bin_index // partition.n_moneyness_bins
        bin_label = partition.bin_label(moneyness_idx, maturity_idx)
        rows.append(
            {
                "surrogate_id": surrogate_id,
                "sampler": sampler,
                "bin_id": bin_index,
                "moneyness_idx": moneyness_idx,
                "maturity_idx": maturity_idx,
                "bin_label": bin_label,
                "is_critical": bin_label in _CRITICAL_BIN_LABELS,
                "n_points": int(report.price["count"][bin_index]),
                "price_mae_mean": float(report.price["mean"][bin_index]),
                "price_mae_p95": float(report.price["p95"][bin_index]),
                "price_mae_p99": float(report.price["p99"][bin_index]),
                "iv_mae_mean": float(report.iv["mean"][bin_index]),  # type: ignore[index]
                "iv_mae_p95": float(report.iv["p95"][bin_index]),  # type: ignore[index]
                "iv_mae_p99": float(report.iv["p99"][bin_index]),  # type: ignore[index]
                "iv_failure_rate": float(report.iv_failure_rate_per_bin[bin_index]),  # type: ignore[index]
            }
        )
    return rows


def _mean_mae_iv_over_critical_bins(report: Report) -> float:
    partition = report.partition
    iv_means = np.asarray(report.iv["mean"])  # type: ignore[index]
    selected: list[float] = []
    for bin_index in range(partition.n_bins):
        moneyness_idx = bin_index % partition.n_moneyness_bins
        maturity_idx = bin_index // partition.n_moneyness_bins
        if partition.bin_label(moneyness_idx, maturity_idx) in _CRITICAL_BIN_LABELS:
            value = float(iv_means[bin_index])
            if np.isfinite(value):
                selected.append(value)
    if not selected:
        return float("nan")
    return float(np.mean(selected))


def _global_mae_iv(report: Report) -> float:
    iv_means = np.asarray(report.iv["mean"])  # type: ignore[index]
    finite = iv_means[np.isfinite(iv_means)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def _relative_change(baseline: float, candidate: float) -> float:
    """Devuelve ``(baseline - candidate) / baseline``.

    En ``improvement_critical``, positivo significa que ``focused`` mejora a
    ``uniform``.
    """
    if not np.isfinite(baseline) or baseline == 0.0 or not np.isfinite(candidate):
        return float("nan")
    return (baseline - candidate) / baseline


def _relative_degradation(baseline: float, candidate: float) -> float:
    """Devuelve ``(candidate - baseline) / baseline``.

    Positivo significa que el candidato empeora; negativo significa que
    mejora.
    """
    if not np.isfinite(baseline) or baseline == 0.0 or not np.isfinite(candidate):
        return float("nan")
    return (candidate - baseline) / baseline


def _build_summary(
    *,
    uniform_id: str,
    focused_id: str,
    uniform_critical: float,
    focused_critical: float,
    uniform_global: float,
    focused_global: float,
    improvement_critical: float,
    global_degradation: float,
    verdict: str,
) -> str:
    lines = [
        "E3 — Muestreo uniforme frente a enfocado.",
        f"  Críticos ({', '.join(_CRITICAL_BIN_LABELS)}):",
        f"    {uniform_id} ({'uniform':<8}) MAE_IV = {uniform_critical:.4e}",
        f"    {focused_id} ({'focused':<8}) MAE_IV = {focused_critical:.4e}",
        f"    mejora relativa en críticos: {improvement_critical:+.2%}",
        "  Global (media nan-safe de medias por bin):",
        f"    {uniform_id}: MAE_IV = {uniform_global:.4e}",
        f"    {focused_id}: MAE_IV = {focused_global:.4e}",
        f"    deterioro global: {global_degradation:+.2%}",
        f"Veredicto (metodologia.md §E3): {verdict}",
    ]
    return "\n".join(lines)
