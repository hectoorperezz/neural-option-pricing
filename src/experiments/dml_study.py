"""E5 — Differential ML con Delta.

Implementa el experimento descrito en ``docs/tasks.md`` y
``docs/metodologia.md``:

    "La comparación principal será
        H-3-small: precio solo, 100k muestras
        H-6-small: precio + Delta, 100k muestras
        H-3: precio solo, 500k muestras"

    "La métrica primaria de E5 será la mejora de `MAE_Delta` de
    H-6-small frente a H-3-small. La restricción de validez es que
    `MAE(C/K)` se mantenga comparable; una mejora de Delta que destruya
    el precio no sería útil. H-3 funciona como referencia de muestra
    grande: si H-6-small se acerca a H-3 usando solo 100k puntos, la
    conclusión será de eficiencia muestral, no solo de calidad de
    Greeks."

    "E5 será positivo fuerte si `MAE_Delta(H-6-small)` mejora al menos
    un 20% frente a `MAE_Delta(H-3-small)` y `MAE(C/K)(H-6-small)` no
    empeora más de un 10% frente a `MAE(C/K)(H-3-small)`. Será positivo
    débil si Delta mejora pero menos de un 20%, siempre que el precio
    siga dentro de ese margen del 10%. Será negativo si Delta no mejora
    o si la mejora de Delta exige sacrificar más de un 10% de precisión
    en precio."

Por eso la clase:

* compara ``small_price`` (H-3-small) y ``small_dml`` (H-6-small);
* acepta ``baseline_large`` (H-3) como referencia observacional;
* desactiva IV porque E5 se centra en Delta y precio;
* aplica los umbrales 20%/10% para el veredicto;
* informa la distancia a H-3 sin usarla para decidir el veredicto.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.evaluation.report import Report
from src.experiments.base import Experiment, ExperimentResult, SurrogateInput


# Roles reconocidos en ``SurrogateInput.labels["role"]``. Los dos primeros
# son obligatorios; el tercero es observacional.
_ROLE_SMALL_PRICE = "small_price"     # H-3-small
_ROLE_SMALL_DML = "small_dml"         # H-6-small
_ROLE_BASELINE_LARGE = "baseline_large"  # H-3
_REQUIRED_ROLES: tuple[str, ...] = (_ROLE_SMALL_PRICE, _ROLE_SMALL_DML)
_OPTIONAL_ROLES: tuple[str, ...] = (_ROLE_BASELINE_LARGE,)
_ALL_ROLES: tuple[str, ...] = _REQUIRED_ROLES + _OPTIONAL_ROLES

# Umbrales definidos en ``metodologia.md`` §E5.
_STRONG_DELTA_IMPROVEMENT: float = 0.20
_MAX_PRICE_DEGRADATION: float = 0.10


@dataclass(frozen=True)
class DMLStudy(Experiment):
    """E5: ``MAE_Delta`` y ``MAE(C/K)`` de H-3-small vs H-6-small (vs H-3)."""

    inputs: tuple[SurrogateInput, ...]

    def __post_init__(self) -> None:
        if not 2 <= len(self.inputs) <= 3:
            raise ValueError(
                "DMLStudy compares two surrogates (small_price, small_dml) "
                "and optionally a third (baseline_large); got "
                f"{len(self.inputs)} inputs"
            )
        roles = [_role_label(item) for item in self.inputs]
        for required in _REQUIRED_ROLES:
            if required not in roles:
                raise ValueError(
                    f"DMLStudy requires an input with labels['role']="
                    f"'{required}'; got roles {roles}"
                )
        for role in roles:
            if role not in _ALL_ROLES:
                raise ValueError(
                    f"unsupported role '{role}'; expected one of {_ALL_ROLES}"
                )
        if len(set(roles)) != len(roles):
            raise ValueError("each role must appear at most once")

    def run(self) -> ExperimentResult:
        reports: dict[str, Report] = {}
        rows: list[dict[str, Any]] = []
        ids_by_role: dict[str, str] = {}

        for surrogate_input in self.inputs:
            role = _role_label(surrogate_input)
            ids_by_role[role] = surrogate_input.surrogate_id

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
                    f"{surrogate_input.surrogate_id}; E5 cannot proceed "
                    "(the test set must carry the `deltas` array)"
                )
            reports[surrogate_input.surrogate_id] = report
            rows.extend(_build_rows(surrogate_input, role, report))

        small_price_id = ids_by_role[_ROLE_SMALL_PRICE]
        small_dml_id = ids_by_role[_ROLE_SMALL_DML]
        baseline_id = ids_by_role.get(_ROLE_BASELINE_LARGE)

        small_price_delta = _global_mean(reports[small_price_id].delta, "mean")
        small_dml_delta = _global_mean(reports[small_dml_id].delta, "mean")
        small_price_price = _global_mean(reports[small_price_id].price, "mean")
        small_dml_price = _global_mean(reports[small_dml_id].price, "mean")

        delta_improvement = _relative_change(small_price_delta, small_dml_delta)
        price_degradation = _relative_change_signed(
            small_price_price, small_dml_price
        )
        verdict = decide_verdict(delta_improvement, price_degradation)

        baseline_delta: float | None = None
        baseline_price: float | None = None
        distance_delta: float | None = None
        distance_price: float | None = None
        if baseline_id is not None:
            baseline_delta = _global_mean(reports[baseline_id].delta, "mean")
            baseline_price = _global_mean(reports[baseline_id].price, "mean")
            distance_delta = _relative_change_signed(baseline_delta, small_dml_delta)
            distance_price = _relative_change_signed(baseline_price, small_dml_price)

        summary = _build_summary(
            small_price_id=small_price_id,
            small_dml_id=small_dml_id,
            baseline_id=baseline_id,
            small_price_delta=small_price_delta,
            small_dml_delta=small_dml_delta,
            small_price_price=small_price_price,
            small_dml_price=small_dml_price,
            baseline_delta=baseline_delta,
            baseline_price=baseline_price,
            delta_improvement=delta_improvement,
            price_degradation=price_degradation,
            distance_delta=distance_delta,
            distance_price=distance_price,
            verdict=verdict,
        )

        return ExperimentResult(
            experiment_id="E5",
            surrogates=tuple(item.surrogate_id for item in self.inputs),
            metric_primary="mejora de MAE_Delta de small_dml frente a small_price",
            table=tuple(rows),
            summary=summary,
            reports=reports,
            verdict=verdict,
        )


def decide_verdict(delta_improvement: float, price_degradation: float) -> str:
    """Aplica los umbrales documentados en ``metodologia.md`` §E5."""
    if not np.isfinite(delta_improvement) or delta_improvement <= 0.0:
        return "negativo"
    if np.isfinite(price_degradation) and price_degradation > _MAX_PRICE_DEGRADATION:
        return "negativo"
    if delta_improvement >= _STRONG_DELTA_IMPROVEMENT:
        return "positivo_fuerte"
    return "positivo_debil"


def _role_label(surrogate_input: SurrogateInput) -> str:
    if surrogate_input.labels is None:
        return ""
    return str(surrogate_input.labels.get("role", ""))


def _loss_label(surrogate_input: SurrogateInput) -> str:
    if surrogate_input.labels is None:
        return ""
    return str(surrogate_input.labels.get("loss", ""))


def _build_rows(
    surrogate_input: SurrogateInput, role: str, report: Report
) -> list[dict[str, Any]]:
    partition = report.partition
    loss_label = _loss_label(surrogate_input)
    rows: list[dict[str, Any]] = []
    for bin_index in range(partition.n_bins):
        moneyness_idx = bin_index % partition.n_moneyness_bins
        maturity_idx = bin_index // partition.n_moneyness_bins
        rows.append(
            {
                "surrogate_id": surrogate_input.surrogate_id,
                "role": role,
                "loss": loss_label,
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


def _global_mean(
    aggregate: dict[str, np.ndarray] | None, statistic: str
) -> float:
    if aggregate is None:
        return float("nan")
    values = np.asarray(aggregate[statistic])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def _relative_change(baseline: float, candidate: float) -> float:
    """``(baseline - candidate) / baseline``; positivo significa mejora."""
    if not np.isfinite(baseline) or baseline == 0.0 or not np.isfinite(candidate):
        return float("nan")
    return (baseline - candidate) / baseline


def _relative_change_signed(baseline: float, candidate: float) -> float:
    """``(candidate - baseline) / baseline``; positivo significa deterioro."""
    if not np.isfinite(baseline) or baseline == 0.0 or not np.isfinite(candidate):
        return float("nan")
    return (candidate - baseline) / baseline


def _build_summary(
    *,
    small_price_id: str,
    small_dml_id: str,
    baseline_id: str | None,
    small_price_delta: float,
    small_dml_delta: float,
    small_price_price: float,
    small_dml_price: float,
    baseline_delta: float | None,
    baseline_price: float | None,
    delta_improvement: float,
    price_degradation: float,
    distance_delta: float | None,
    distance_price: float | None,
    verdict: str,
) -> str:
    lines = [
        "E5 — Differential ML con Delta.",
        f"  {small_price_id} (small_price): "
        f"MAE_Delta={small_price_delta:.4e}, MAE(C/K)={small_price_price:.4e}",
        f"  {small_dml_id} (small_dml):     "
        f"MAE_Delta={small_dml_delta:.4e}, MAE(C/K)={small_dml_price:.4e}",
        f"  Delta improvement (small_dml vs small_price): "
        f"{delta_improvement:+.2%}",
        f"  Price degradation (small_dml vs small_price): "
        f"{price_degradation:+.2%}",
    ]
    if baseline_id is not None:
        lines.append(
            f"  {baseline_id} (baseline_large): "
            f"MAE_Delta={baseline_delta:.4e}, MAE(C/K)={baseline_price:.4e}"
        )
        if distance_delta is not None:
            lines.append(
                f"  Distancia small_dml vs baseline: "
                f"Delta={distance_delta:+.2%}, precio={distance_price:+.2%}"
            )
    lines.append(f"Veredicto (metodologia.md §E5): {verdict}")
    return "\n".join(lines)
