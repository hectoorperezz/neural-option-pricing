"""Plano de compromiso precio <-> Delta para E2.

Un punto por activación en el plano (error de precio, error de Delta). Abajo a
la izquierda = mejor en ambas métricas. Faceta por familia (BS | Heston) para
leer el compromiso dentro de cada una. Identidad visual común a través de
``paper_style`` (paleta amarillo/negro, tinta serif, acento amarillo).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from paper_style import ACT_COLOR, ACT_LABEL, PALETTE, apply_paper_style

REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "docs" / "latex" / "assets" / "e2_tradeoff.png"

apply_paper_style()

panels = [
    ("Black-Scholes", REPO / "results" / "metrics" / "e2_bs.csv"),
    ("Heston", REPO / "results" / "metrics" / "e2_heston.csv"),
]

fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.5))

for ax, (family, csv) in zip(axes, panels):
    df = pd.read_csv(csv)
    agg = df.groupby("activation").agg(
        price=("price_mae_mean", "mean"),
        delta=("delta_mae_mean", "mean"),
    )
    for act, row in agg.iterrows():
        ax.scatter(row["price"], row["delta"], s=150, color=ACT_COLOR[act],
                   edgecolors=PALETTE["uniblack"], linewidths=0.9, zorder=3)
        ax.annotate(ACT_LABEL[act], (row["price"], row["delta"]),
                    textcoords="offset points", xytext=(9, 4),
                    fontsize=10, color=PALETTE["uniblack"])

    ax.set_title(family, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(r"MAE$(C/K)$  (error de precio)", fontsize=10.5)

    # Ticks compactos con factor común para evitar el solape del eje X.
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.xaxis.get_offset_text().set_fontsize(8.5)
    ax.yaxis.get_offset_text().set_fontsize(8.5)
    ax.tick_params(labelsize=9)

    # Holgura para que las etiquetas no se salgan del marco.
    xmin, xmax = agg["price"].min(), agg["price"].max()
    ymin, ymax = agg["delta"].min(), agg["delta"].max()
    ax.set_xlim(xmin - 0.20 * (xmax - xmin), xmax + 0.32 * (xmax - xmin))
    ax.set_ylim(ymin - 0.14 * (ymax - ymin), ymax + 0.20 * (ymax - ymin))

axes[0].set_ylabel(r"MAE$_\Delta$  (error de Delta)", fontsize=10.5)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
