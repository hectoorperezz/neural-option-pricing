"""Plano de compromiso precio <-> Delta para E5.

Tres surrogates en el plano (error de precio, error de Delta), promediados sobre
los 25 bins: H-3-small (precio, 5M), H-6-small (precio+Delta, 5M) y H-3
(precio, 25M). Abajo a la izquierda = mejor en ambas métricas. Identidad visual
del paper vía paper_style.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from paper_style import apply_paper_style, PALETTE

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "results" / "metrics" / "e5_table.csv"
OUT = REPO / "docs" / "latex" / "assets" / "e5_tradeoff.png"

apply_paper_style()

df = pd.read_csv(CSV)
agg = df.groupby("role").agg(price=("price_mae_mean", "mean"),
                             delta=("delta_mae_mean", "mean"))

# (rol, etiqueta, color, desplazamiento de la etiqueta)
points = [
    ("small_price", "H-3-small (precio, 5M)", PALETTE["orange"], (10, 2)),
    ("small_dml",   "H-6-small (precio+$\\Delta$, 5M)", PALETTE["green"], (10, -4)),
    ("baseline_large", "H-3 (precio, 25M)", PALETTE["uniblack"], (10, 8)),
]

fig, ax = plt.subplots(figsize=(6.6, 5.0))
for role, label, color, off in points:
    x, y = agg.loc[role, "price"], agg.loc[role, "delta"]
    ha = "right" if off[0] < 0 else "left"
    ax.scatter(x, y, s=150, color=color, edgecolors=PALETTE["uniblack"],
               linewidths=0.9, zorder=3)
    ax.annotate(label, (x, y), textcoords="offset points", xytext=off,
                fontsize=9.5, color=PALETTE["uniblack"], ha=ha)

ax.set_xlabel(r"MAE$(C/K)$  (error de precio)", fontsize=10.5)
ax.set_ylabel(r"MAE$_\Delta$  (error de Delta)", fontsize=10.5)
ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.xaxis.get_offset_text().set_fontsize(8.5)
ax.yaxis.get_offset_text().set_fontsize(8.5)
ax.tick_params(labelsize=9)

xmin, xmax = agg["price"].min(), agg["price"].max()
ymin, ymax = agg["delta"].min(), agg["delta"].max()
ax.set_xlim(xmin - 0.22 * (xmax - xmin), xmax + 0.30 * (xmax - xmin))
ax.set_ylim(ymin - 0.18 * (ymax - ymin), ymax + 0.20 * (ymax - ymin))

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
