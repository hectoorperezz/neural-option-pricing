"""E1 — Precio frente a volatilidad implícita.

Implementa el experimento descrito en ``docs/tasks.md`` y
``docs/metodologia.md``:

    "E1 es observacional. No requiere entrenar nada nuevo, simplemente
    recalcula métricas sobre un surrogate que ya está entrenado para
    otros experimentos."

    "La métrica primaria de E1 no será un único promedio, sino la
    discrepancia por bin entre `MAE(C/K)` y `MAE_IV`."

    "El entregable debe incluir tabla por bin con `MAE(C/K)`, `MAE_IV`,
    Vega media o proxy de Vega, percentiles altos y tasa de fallos de
    inversión IV, además de heatmaps separados para precio e IV."

    "No fijamos clasificación fuerte/débil para E1."

Por eso la clase:

* fuerza ``compute_iv=True``, porque IV es la métrica central;
* calcula una Vega proxy como ``BS-Vega(target IV)``, agregada por bin;
* emite precio, IV, Vega proxy, tasa de fallo IV y ``iv_to_price_ratio``;
* genera un resumen observacional, sin veredicto fuerte/débil.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.evaluation.metrics import (
    aggregate_by_bin,
    invert_implied_volatility_call,
)
from src.evaluation.report import Report
from src.experiments.base import Experiment, ExperimentResult, SurrogateInput
from src.solvers import BlackScholesSolver


@dataclass(frozen=True)
class PriceVsIVStudy(Experiment):
    """E1: discrepancia entre ``MAE(C/K)`` y ``MAE_IV`` por bin."""

    inputs: tuple[SurrogateInput, ...]

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("PriceVsIVStudy needs at least one SurrogateInput")

    def run(self) -> ExperimentResult:
        rows: list[dict[str, Any]] = []
        reports: dict[str, Report] = {}

        for surrogate_input in self.inputs:
            report = surrogate_input.evaluator.evaluate(
                surrogate=surrogate_input.model,
                dataset=surrogate_input.dataset,
                bin_id=surrogate_input.bin_id,
                compute_iv=True,
                surrogate_id=surrogate_input.surrogate_id,
            )
            if report.iv is None or report.iv_failure_rate_per_bin is None:
                raise RuntimeError(
                    f"BinEvaluator returned no IV aggregates for "
                    f"{surrogate_input.surrogate_id}; E1 cannot proceed"
                )

            vega_per_bin = _vega_proxy_per_bin(
                surrogate_input,
                report,
            )
            rows.extend(
                _build_rows(
                    surrogate_input.surrogate_id,
                    report,
                    vega_per_bin,
                )
            )
            reports[surrogate_input.surrogate_id] = report

        summary = _build_summary(rows)
        return ExperimentResult(
            experiment_id="E1",
            surrogates=tuple(item.surrogate_id for item in self.inputs),
            metric_primary="discrepancia entre MAE(C/K) y MAE_IV por bin",
            table=tuple(rows),
            summary=summary,
            reports=reports,
        )


def _vega_proxy_per_bin(
    surrogate_input: SurrogateInput,
    report: Report,
) -> np.ndarray:
    """Calcula la Vega proxy media por bin.

    Usamos la Vega Black-Scholes evaluada en la IV del precio objetivo, es
    decir, del solver y no de la red. Es una proxy común para BS y Heston
    porque la IV siempre se recupera contra Black-Scholes.
    """
    raw_inputs = surrogate_input.dataset.raw_inputs.detach().cpu().numpy()
    moneyness = raw_inputs[:, 0].astype(np.float64)
    maturity = raw_inputs[:, 1].astype(np.float64)
    rate = raw_inputs[:, 2].astype(np.float64)
    target_prices = (
        surrogate_input.dataset.prices.detach().cpu().numpy().reshape(-1).astype(np.float64)
    )

    inverter = (
        surrogate_input.evaluator.iv_inverter
        if surrogate_input.evaluator.iv_inverter is not None
        else None
    )
    target_iv, ok_target = invert_implied_volatility_call(
        prices=target_prices,
        moneyness=moneyness,
        maturity=maturity,
        rate=rate,
        inverter=inverter,
        workers=surrogate_input.evaluator.iv_workers,
        progress=surrogate_input.evaluator.iv_progress,
    )

    n_points = target_prices.shape[0]
    vega_per_point = np.full(n_points, np.nan, dtype=np.float64)
    if np.any(ok_target):
        bs = BlackScholesSolver()
        vega_per_point[ok_target] = bs.call_vega(
            spot=moneyness[ok_target],
            strike=1.0,
            maturity=maturity[ok_target],
            rate=rate[ok_target],
            volatility=target_iv[ok_target],
            dividend_yield=0.0,
        )

    bin_id = _resolve_bin_id_for_dataset(surrogate_input, moneyness, maturity)
    aggregate = aggregate_by_bin(
        vega_per_point, bin_id, surrogate_input.evaluator.partition.n_bins
    )
    return aggregate["mean"]


def _resolve_bin_id_for_dataset(
    surrogate_input: SurrogateInput,
    moneyness: np.ndarray,
    maturity: np.ndarray,
) -> np.ndarray:
    if surrogate_input.bin_id is not None:
        return np.asarray(surrogate_input.bin_id, dtype=np.int64)
    assigned, _, _ = surrogate_input.evaluator.partition.assign(moneyness, maturity)
    return assigned


def _build_rows(
    surrogate_id: str,
    report: Report,
    vega_proxy_per_bin: np.ndarray,
) -> list[dict[str, Any]]:
    partition = report.partition
    rows: list[dict[str, Any]] = []
    for bin_index in range(partition.n_bins):
        moneyness_idx = bin_index % partition.n_moneyness_bins
        maturity_idx = bin_index // partition.n_moneyness_bins
        price_mean = float(report.price["mean"][bin_index])
        iv_mean = float(report.iv["mean"][bin_index])  # type: ignore[index]
        ratio = _discrepancy_ratio(iv_mean, price_mean)

        rows.append(
            {
                "surrogate_id": surrogate_id,
                "bin_id": bin_index,
                "moneyness_idx": moneyness_idx,
                "maturity_idx": maturity_idx,
                "bin_label": partition.bin_label(moneyness_idx, maturity_idx),
                "n_points": int(report.price["count"][bin_index]),
                "price_mae_mean": price_mean,
                "price_mae_p95": float(report.price["p95"][bin_index]),
                "price_mae_p99": float(report.price["p99"][bin_index]),
                "iv_mae_mean": iv_mean,
                "iv_mae_p95": float(report.iv["p95"][bin_index]),  # type: ignore[index]
                "iv_mae_p99": float(report.iv["p99"][bin_index]),  # type: ignore[index]
                "iv_failure_rate": float(report.iv_failure_rate_per_bin[bin_index]),  # type: ignore[index]
                "vega_proxy_mean": float(vega_proxy_per_bin[bin_index]),
                "iv_to_price_ratio": ratio,
            }
        )
    return rows


def _discrepancy_ratio(iv_mae: float, price_mae: float) -> float:
    if not np.isfinite(iv_mae) or not np.isfinite(price_mae) or price_mae <= 0.0:
        return float("nan")
    return iv_mae / price_mae


def _build_summary(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No data available."

    finite_rows = [row for row in rows if np.isfinite(row["iv_to_price_ratio"])]
    if not finite_rows:
        return (
            "No bin has both finite price MAE and finite IV MAE; "
            "IV inversion may have failed everywhere."
        )

    finite_rows.sort(key=lambda row: row["iv_to_price_ratio"], reverse=True)
    top = finite_rows[:3]
    failure_rates = [
        row["iv_failure_rate"]
        for row in rows
        if np.isfinite(row["iv_failure_rate"])
    ]
    avg_failure = (
        float(np.mean(failure_rates)) if failure_rates else float("nan")
    )

    lines: list[str] = [
        "E1 — Discrepancia entre MAE(C/K) y MAE_IV por bin (observacional).",
        "Bins con mayor iv_to_price_ratio (peor traducción precio -> IV):",
    ]
    for row in top:
        lines.append(
            f"  {row['surrogate_id']} :: {row['bin_label']} "
            f"ratio={row['iv_to_price_ratio']:.2f} "
            f"price_mae={row['price_mae_mean']:.2e} "
            f"iv_mae={row['iv_mae_mean']:.2e}"
        )
    if np.isfinite(avg_failure):
        lines.append(f"IV failure rate medio por bin: {avg_failure:.2%}")
    return "\n".join(lines)
