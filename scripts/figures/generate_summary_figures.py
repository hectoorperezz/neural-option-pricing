"""Genera figuras de síntesis a partir de los CSV de experimentos.

Los scripts de experimento escriben CSVs detallados por bin. Este script lee
esos CSVs y produce figuras más directas para paper y presentación:

* E1: scatter de Vega proxy frente a amplificación de IV;
* E2: ranking de activaciones;
* E3: heatmap de mejora relativa;
* E5: heatmap de mejora relativa y comparación precio/Delta.

Uso típico::

    python scripts/figures/generate_summary_figures.py
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "proyecto_final_mpl_cache"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "proyecto_final_xdg_cache"),
)

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


MONEYNESS_LABELS = ("deep OTM", "OTM", "ATM", "ITM", "deep ITM")
MATURITY_LABELS = ("weekly", "short", "medium-short", "medium", "long")

ACTIVATION_ORDER = ("relu", "softplus", "swish", "tanh")
ACTIVATION_LABELS = {
    "relu": "ReLU",
    "softplus": "Softplus",
    "swish": "Swish",
    "tanh": "tanh",
}


def parse_args() -> argparse.Namespace:
    """Parser CLI; los ``help`` describen cada flag."""
    parser = argparse.ArgumentParser(
        description="Genera figuras de síntesis desde los CSV de métricas."
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("results/metrics"),
        help="Directorio que contiene e1_table.csv ... e5_table.csv.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directorio raíz donde se escribirán las figuras de síntesis.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    """Lee un CSV como lista de dicts ``{columna: cadena}``."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    """Parsea ``row[key]`` a ``float``; cadena vacía se traduce a ``NaN``."""
    value = row[key]
    if value == "":
        return float("nan")
    return float(value)


def mean(values: Iterable[float]) -> float:
    """Media ignorando ``NaN``; devuelve ``NaN`` si no hay valores finitos."""
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def group_mean(rows: Iterable[dict[str, str]], group_key: str, value_key: str) -> dict[str, float]:
    """Agrupa por ``group_key`` y devuelve la media de ``value_key`` por grupo."""
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row[group_key], []).append(as_float(row, value_key))
    return {key: mean(values) for key, values in grouped.items()}


def grid_by_surrogate(
    rows: Iterable[dict[str, str]],
    surrogate_id: str,
    value_key: str,
) -> np.ndarray:
    """Vuelca ``value_key`` de un surrogate a la malla 5x5 ``(maturity, moneyness)``."""
    grid = np.full((len(MATURITY_LABELS), len(MONEYNESS_LABELS)), np.nan)
    for row in rows:
        if row["surrogate_id"] != surrogate_id:
            continue
        maturity_idx = int(row["maturity_idx"])
        moneyness_idx = int(row["moneyness_idx"])
        grid[maturity_idx, moneyness_idx] = as_float(row, value_key)
    return grid


def save_relative_heatmap(
    grid: np.ndarray,
    output: Path,
    *,
    title: str,
    cbar_label: str,
) -> Path:
    """Guarda un heatmap centrado en 0 con anotaciones en porcentaje."""
    output.parent.mkdir(parents=True, exist_ok=True)
    finite = grid[np.isfinite(grid)]
    max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    image = ax.imshow(grid, cmap="RdYlGn", norm=norm, aspect="auto", origin="upper")
    ax.set_xticks(range(len(MONEYNESS_LABELS)))
    ax.set_xticklabels(MONEYNESS_LABELS, rotation=25, ha="right")
    ax.set_yticks(range(len(MATURITY_LABELS)))
    ax.set_yticklabels(MATURITY_LABELS)
    ax.set_xlabel("Moneyness bin")
    ax.set_ylabel("Maturity bin")
    ax.set_title(title)

    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            value = grid[row, col]
            if not np.isfinite(value):
                continue
            ax.text(
                col,
                row,
                f"{value:+.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    fig.colorbar(image, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_e1_vega_scatter(metrics_dir: Path, figures_dir: Path) -> Path:
    """E1: amplificación de IV (``MAE_IV/MAE(C/K)``) frente a Vega proxy."""
    rows = read_rows(metrics_dir / "e1_table.csv")
    output = figures_dir / "e1" / "vega_proxy_vs_iv_ratio.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = {"BS-3": "#2f80ed", "H-3": "#d64545"}
    for surrogate_id in ("BS-3", "H-3"):
        subset = [row for row in rows if row["surrogate_id"] == surrogate_id]
        ax.scatter(
            [as_float(row, "vega_proxy_mean") for row in subset],
            [as_float(row, "iv_to_price_ratio") for row in subset],
            s=52,
            alpha=0.78,
            label=surrogate_id,
            color=colors[surrogate_id],
            edgecolors="white",
            linewidths=0.6,
        )

    top = sorted(rows, key=lambda row: as_float(row, "iv_to_price_ratio"), reverse=True)[:3]
    for row in top:
        ax.annotate(
            row["bin_label"].replace("_", " "),
            (as_float(row, "vega_proxy_mean"), as_float(row, "iv_to_price_ratio")),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Vega proxy mean")
    ax.set_ylabel("MAE_IV / MAE(C/K)")
    ax.set_title("E1: IV error amplification in low-Vega regions")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_e2_activation_ranking(metrics_dir: Path, figures_dir: Path) -> Path:
    """E2: ranking de activaciones por ``MAE_Delta`` medio (BS y Heston)."""
    bs_rows = read_rows(metrics_dir / "e2_bs.csv")
    heston_rows = read_rows(metrics_dir / "e2_heston.csv")
    output = figures_dir / "e2" / "activation_delta_ranking.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    bs_delta = group_mean(bs_rows, "activation", "delta_mae_mean")
    heston_delta = group_mean(heston_rows, "activation", "delta_mae_mean")
    x = np.arange(len(ACTIVATION_ORDER))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.bar(
        x - width / 2,
        [bs_delta[key] for key in ACTIVATION_ORDER],
        width,
        label="Black-Scholes",
        color="#2f80ed",
    )
    ax.bar(
        x + width / 2,
        [heston_delta[key] for key in ACTIVATION_ORDER],
        width,
        label="Heston",
        color="#d64545",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([ACTIVATION_LABELS[key] for key in ACTIVATION_ORDER])
    ax.set_ylabel("Mean MAE_Delta")
    ax.set_title("E2: activation ranking by Delta error")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_e3_relative_improvement(metrics_dir: Path, figures_dir: Path) -> Path:
    """E3: mejora relativa de ``MAE_IV`` de H-5 respecto a H-3 por bin."""
    rows = read_rows(metrics_dir / "e3_table.csv")
    h3 = grid_by_surrogate(rows, "H-3", "iv_mae_mean")
    h5 = grid_by_surrogate(rows, "H-5", "iv_mae_mean")
    improvement = (h3 - h5) / h3
    return save_relative_heatmap(
        improvement,
        figures_dir / "e3" / "H5_vs_H3_iv_relative_improvement.png",
        title="E3: focused sampler relative improvement in MAE_IV",
        cbar_label="(H-3 - H-5) / H-3",
    )


def plot_e5_relative_improvement(metrics_dir: Path, figures_dir: Path) -> Path:
    """E5: mejora relativa de ``MAE_Delta`` del DML respecto al price-only."""
    rows = read_rows(metrics_dir / "e5_table.csv")
    price_only = grid_by_surrogate(rows, "H-3-small", "delta_mae_mean")
    dml = grid_by_surrogate(rows, "H-6-small", "delta_mae_mean")
    improvement = (price_only - dml) / price_only
    return save_relative_heatmap(
        improvement,
        figures_dir / "e5" / "H6small_vs_H3small_delta_relative_improvement.png",
        title="E5: DML relative improvement in MAE_Delta",
        cbar_label="(H-3-small - H-6-small) / H-3-small",
    )


def plot_e5_metric_comparison(metrics_dir: Path, figures_dir: Path) -> Path:
    """E5: barras comparando precio y Delta para los tres surrogates de E5."""
    rows = read_rows(metrics_dir / "e5_table.csv")
    price = group_mean(rows, "surrogate_id", "price_mae_mean")
    delta = group_mean(rows, "surrogate_id", "delta_mae_mean")
    surrogates = ("H-3-small", "H-6-small", "H-3")
    labels = ("small price", "small DML", "large price")
    output = figures_dir / "e5" / "e5_price_delta_summary.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    colors = ("#7a7a7a", "#2e8b57", "#2f80ed")
    axes[0].bar(labels, [price[key] for key in surrogates], color=colors)
    axes[0].set_title("Price error")
    axes[0].set_ylabel("Mean MAE(C/K)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(labels, [delta[key] for key in surrogates], color=colors)
    axes[1].set_title("Delta error")
    axes[1].set_ylabel("Mean MAE_Delta")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("E5: price-only vs differential training", weight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main() -> None:
    """Entrada del script: genera las 5 figuras de síntesis (E1-E5)."""
    args = parse_args()
    metrics_dir = args.metrics_dir
    figures_dir = args.figures_dir

    written: list[Path] = []
    written.append(plot_e1_vega_scatter(metrics_dir, figures_dir))
    written.append(plot_e2_activation_ranking(metrics_dir, figures_dir))
    written.append(plot_e3_relative_improvement(metrics_dir, figures_dir))
    written.append(plot_e5_relative_improvement(metrics_dir, figures_dir))
    written.append(plot_e5_metric_comparison(metrics_dir, figures_dir))

    for path in written:
        print(f"Wrote figure: {path}")


if __name__ == "__main__":
    main()
