"""Resumen de E6: MAE(C/K) y MAE(Delta) por variante de profundidad/scheduler.

Barras de las cuatro variantes (H-3 baseline, H-7-shallow, H-8-deep y
H-9-lr-schedule) promediando sobre los 25 bins del test balanceado, a partir de
``results/metrics/e6_table.csv``. El baseline H-3 se resalta en dorado. Es la
misma figura que el notebook de entrega, con la identidad visual del paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd

from paper_style import PALETTE, apply_paper_style

REPO = Path(__file__).resolve().parent.parent.parent
CSV = REPO / "results" / "metrics" / "e6_table.csv"
OUT_DIRS = (
    REPO / "results" / "figures" / "e6",
    REPO / "docs" / "latex" / "assets",
)

apply_paper_style()

df = pd.read_csv(CSV)
agg = (
    df.groupby(["surrogate_id", "role"])
    .agg(price=("price_mae_mean", "mean"), delta=("delta_mae_mean", "mean"))
    .reset_index()
    .sort_values("price")
)
colors = [PALETTE["gold"] if r == "baseline" else PALETTE["uniblack"] for r in agg["role"]]
xpos = range(len(agg))

fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))
for a, col, title in (
    (ax[0], "price", r"MAE$(C/K)$ medio por bin"),
    (ax[1], "delta", r"MAE$_\Delta$ medio por bin"),
):
    a.bar(xpos, agg[col], color=colors, edgecolor=PALETTE["uniblack"], linewidth=0.6)
    a.set_title(title, fontsize=11, fontweight="bold", pad=8)
    a.set_xticks(list(xpos))
    a.set_xticklabels(agg["surrogate_id"], rotation=20, ha="right", fontsize=8.5)
    a.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    a.yaxis.get_offset_text().set_fontsize(8)
    a.grid(axis="x", visible=False)
fig.tight_layout()

for out_dir in OUT_DIRS:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "e6_summary.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote", path)
plt.close(fig)
