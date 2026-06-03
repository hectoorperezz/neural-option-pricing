"""Heatmap de mejora relativa de MAE_IV entre H-3 (uniforme) y H-5 (enfocado).

Cada celda muestra (MAE_IV^{H-3} - MAE_IV^{H-5}) / MAE_IV^{H-3} por bin: positivo
(verde) = el muestreo enfocado reduce el error de IV; negativo (rojo) = lo
empeora. Colormap divergente en la paleta del paper, centrado en cero.
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
from matplotlib.ticker import PercentFormatter

from paper_style import (
    apply_paper_style, diverging_cmap, PALETTE,
    MONEYNESS_LABELS, MATURITY_LABELS,
)

REPO = Path(__file__).resolve().parent.parent.parent
CSV = REPO / "results" / "metrics" / "e3_table.csv"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--column", default="iv_mae_mean")
parser.add_argument("--metric-label", default="MAE$_{IV}$")
parser.add_argument("--out", type=Path,
                    default=REPO / "results" / "figures" / "e3" / "e3_diff.png")
args = parser.parse_args()
COLUMN, METRIC_LABEL, OUT = args.column, args.metric_label, args.out

apply_paper_style()

df = pd.read_csv(CSV)
u = df[df.surrogate_id == "H-3"].set_index("bin_label")
f = df[df.surrogate_id == "H-5"].set_index("bin_label")

grid = np.full((5, 5), np.nan)
for label, row in u.iterrows():
    i, j = int(row["maturity_idx"]), int(row["moneyness_idx"])
    rel = (row[COLUMN] - f.loc[label, COLUMN]) / row[COLUMN]
    grid[i, j] = rel

vmax = np.nanmax(np.abs(grid))
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
cmap = diverging_cmap()

fig, ax = plt.subplots(figsize=(5.8, 5.0))
ax.grid(False)
image = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto", origin="upper")

ax.set_xticks(range(5)); ax.set_xticklabels(MONEYNESS_LABELS, fontsize=8.5)
ax.set_yticks(range(5)); ax.set_yticklabels(MATURITY_LABELS, fontsize=8.5)
ax.set_xlabel("Moneyness $m = S/K$", fontsize=9.5)
ax.set_ylabel("Vencimiento $T$", fontsize=9.5)
ax.set_title(f"H-5 frente a H-3 — mejora relativa de {METRIC_LABEL}", fontsize=10.5)

for i in range(5):
    for j in range(5):
        v = grid[i, j]
        if not np.isfinite(v):
            continue
        rgba = cmap(norm(v))
        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        color = "white" if lum < 0.5 else PALETTE["uniblack"]
        ax.text(j, i, f"{v * 100:+.0f}%", ha="center", va="center",
                fontsize=8.5, color=color)

cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04,
                    format=PercentFormatter(xmax=1.0, decimals=0))
cbar.ax.tick_params(labelsize=8)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
