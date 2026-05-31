"""Scatter MAE_IV vs MAE(C/K) por bin para E1.

Cada punto es un bin del test balanceado. Eje X: error de precio MAE(C/K).
Eje Y: error de volatilidad implícita MAE_IV (ambos log). El color codifica la
Vega proxy del bin y las diagonales discontinuas marcan razones constantes
MAE_IV/MAE(C/K). Marcadores por surrogate: círculo H-3, triángulo BS-3.

Sin título dentro de la figura: la interpretación va en el caption del paper.
Paleta alineada con paper.tex; colormap que no llega al blanco para que todo
punto sea visible sobre fondo claro.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_style import apply_paper_style, PALETTE

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "results" / "metrics" / "e1_table.csv"
OUT = REPO / "docs" / "latex" / "assets" / "e1_scatter.png"

apply_paper_style()
UNIBLACK = PALETTE["uniblack"]
# Negro -> ámbar -> oro. No alcanza el blanco, así los puntos de Vega alta
# siguen siendo visibles sobre fondo claro.
VEGA_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "vega", ["#1A1A1A", "#6E5600", "#B89200", "#FFD100"]
)

frame = pd.read_csv(CSV)
fig, ax = plt.subplots(figsize=(6.8, 5.2))

# Diagonales de razón constante MAE_IV = k * MAE(C/K).
x_line = np.array([8e-5, 1.2e-2])
for k, label in [(1, "1:1"), (10, r"$\times 10$"),
                 (100, r"$\times 100$"), (1000, r"$\times 1000$")]:
    ax.plot(x_line, k * x_line, ls="--", lw=0.8, color="#BBBBBB", zorder=1)
    ax.text(x_line[1], k * x_line[1], f" {label}", fontsize=8,
            color="#8A8A8A", va="center", ha="left")

vega_all = frame["vega_proxy_mean"].to_numpy()
norm = mcolors.LogNorm(vmin=np.nanmin(vega_all), vmax=np.nanmax(vega_all))

markers = {"H-3": "o", "BS-3": "^"}
sc = None
for surrogate, marker in markers.items():
    sub = frame[frame["surrogate_id"] == surrogate]
    sc = ax.scatter(
        sub["price_mae_mean"], sub["iv_mae_mean"],
        c=sub["vega_proxy_mean"], cmap=VEGA_CMAP, norm=norm,
        marker=marker, s=85, edgecolors=UNIBLACK, linewidths=0.7,
        label=surrogate, zorder=3,
    )

# Anotar el peor bin de cada surrogate, con desplazamientos que no chocan.
offsets = {"H-3": (-2, 10), "BS-3": (8, -14)}
for surrogate in ("H-3", "BS-3"):
    sub = frame[frame["surrogate_id"] == surrogate]
    worst = sub.loc[sub["iv_to_price_ratio"].idxmax()]
    ax.annotate(
        f'{surrogate}: {worst["bin_label"].replace("_", " ")}',
        (worst["price_mae_mean"], worst["iv_mae_mean"]),
        textcoords="offset points", xytext=offsets[surrogate],
        fontsize=8, color=UNIBLACK,
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"MAE$(C/K)$", fontsize=11)
ax.set_ylabel(r"MAE$_{\mathrm{IV}}$", fontsize=11)
ax.grid(True, which="both", ls=":", lw=0.4, color="#E2E2E2")
ax.legend(title="Surrogate", fontsize=9, title_fontsize=9, loc="lower right")

cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Vega proxy media del bin", fontsize=9)
cbar.ax.tick_params(labelsize=8)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
