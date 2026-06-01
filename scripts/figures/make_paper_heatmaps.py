"""Genera heatmaps 5x5 legibles y en español para el paper.

Lee un CSV de métricas por bin (los de ``results/metrics/``) y produce un
heatmap por surrogate y columna, con anotaciones grandes, etiquetas en
español y la paleta institucional (amarillo Hespérides -> marrón oscuro).

A diferencia de los heatmaps de ``results/figures/`` (matplotlib por defecto,
viridis, fuente pequeña, rótulos en inglés), estos están pensados para
reducirse a media columna en el paper sin perder legibilidad.

Uso:
    python scripts/figures/make_paper_heatmaps.py \
        --csv results/metrics/e2_heston.csv \
        --column delta_mae_mean \
        --surrogate H-3 \
        --metric-label "MAE$_\\Delta$" \
        --out docs/latex/assets/e2_h3_delta.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_style import apply_paper_style, heatmap_cmap, PALETTE

apply_paper_style()
UNIBLACK = PALETTE["uniblack"]
PAPER_CMAP = heatmap_cmap()

# Terminología técnica en inglés, igual que la figura de bins del paper.
MONEYNESS_LABELS = ("Deep\nOTM", "OTM", "ATM", "ITM", "Deep\nITM")
MATURITY_LABELS = ("Weekly", "Short", "Medium-\nshort", "Medium", "Long")


def _fmt(value: float) -> str:
    """Formato compacto: mantisa con dos decimales y exponente, p. ej. 1.57e0."""
    if not np.isfinite(value):
        return ""
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0 ** exponent)
    return f"{mantissa:.2f}e{exponent}"


def make_heatmap(
    csv: Path,
    column: str,
    surrogate: str,
    out: Path,
    metric_label: str,
    log_color: bool = False,
    suffix: str = "(media por bin)",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    frame = pd.read_csv(csv)
    frame = frame[frame["surrogate_id"] == surrogate]
    if frame.empty:
        raise SystemExit(f"no rows for surrogate {surrogate!r} in {csv}")

    grid = np.full((5, 5), np.nan)
    for _, row in frame.iterrows():
        grid[int(row["maturity_idx"]), int(row["moneyness_idx"])] = row[column]

    finite = grid[np.isfinite(grid)]
    lo = finite.min() if vmin is None else vmin
    hi = finite.max() if vmax is None else vmax
    if log_color:
        norm = mcolors.LogNorm(vmin=lo, vmax=hi)
    else:
        norm = mcolors.Normalize(vmin=lo, vmax=hi)

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.grid(False)  # sin rejilla sobre el heatmap (el estilo global la activa)
    masked = np.ma.masked_invalid(grid)
    cmap = PAPER_CMAP.copy()
    cmap.set_bad("#dddddd")
    image = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto", origin="upper")

    ax.set_xticks(range(5))
    ax.set_xticklabels(MONEYNESS_LABELS, fontsize=8.5)
    ax.set_yticks(range(5))
    ax.set_yticklabels(MATURITY_LABELS, fontsize=8.5)
    ax.set_xlabel("Moneyness $m = S/K$", fontsize=9.5)
    ax.set_ylabel("Vencimiento $T$", fontsize=9.5)
    title = f"{surrogate} — {metric_label}"
    if suffix:
        title += f" {suffix}"
    ax.set_title(title, fontsize=10.5)

    # Umbral de luminancia para decidir color del texto en cada casilla.
    for i in range(5):
        for j in range(5):
            value = grid[i, j]
            if not np.isfinite(value):
                continue
            rgba = cmap(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if luminance < 0.5 else UNIBLACK
            ax.text(
                j, i, _fmt(value),
                ha="center", va="center",
                fontsize=8.5, color=text_color,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--column", required=True)
    parser.add_argument("--surrogate", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metric-label", default="MAE")
    parser.add_argument("--log-color", action="store_true")
    parser.add_argument("--suffix", default="(media por bin)")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    args = parser.parse_args()
    path = make_heatmap(
        args.csv, args.column, args.surrogate, args.out,
        args.metric_label, args.log_color, args.suffix,
        vmin=args.vmin, vmax=args.vmax,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
