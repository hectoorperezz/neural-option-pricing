"""Figura de eficiencia E4: tiempo y speedup frente al tamaño de lote.

Panel izquierdo: tiempo mediano por lote (log-log) del solver de Heston y del
surrogate en CPU y GPU, con banda p25-p75. Panel derecho: speedup
(tiempo_solver / tiempo_surrogate) por lote, en CPU y GPU. Estilo del paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd

from paper_style import apply_paper_style, PALETTE

REPO = Path(__file__).resolve().parent.parent.parent
CSV = REPO / "results" / "metrics" / "e4_table.csv"
OUT = REPO / "results" / "figures" / "e4" / "e4_timing.png"

apply_paper_style()

df = pd.read_csv(CSV)
cpu = df[df.device == "cpu"].sort_values("batch_size")
gpu = df[df.device == "cuda"].sort_values("batch_size")
batches = cpu["batch_size"].to_numpy()

C_SOLVER = PALETTE["uniblack"]   # solver: negro (línea base lenta)
C_CPU = PALETTE["gold"]          # surrogate CPU: oro oscuro
C_GPU = PALETTE["uniyellow"]     # surrogate GPU: amarillo Hespérides (el más rápido)
EDGE = PALETTE["uniblack"]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.4))

# --- Panel izquierdo: tiempo vs lote -----------------------------------------
def time_curve(ax, x, med, p25, p75, color, label):
    ax.plot(x, med, "o-", color=color, lw=1.8, ms=6, label=label, zorder=3,
            markeredgecolor=EDGE, markeredgewidth=0.6)
    ax.fill_between(x, p25, p75, color=color, alpha=0.18, zorder=1)

time_curve(axL, batches, cpu["solver_median_s"], cpu["solver_p25_s"],
           cpu["solver_p75_s"], C_SOLVER, "Solver (Heston)")
time_curve(axL, batches, cpu["surrogate_median_s"], cpu["surrogate_p25_s"],
           cpu["surrogate_p75_s"], C_CPU, "Surrogate (CPU)")
time_curve(axL, batches, gpu["surrogate_median_s"], gpu["surrogate_p25_s"],
           gpu["surrogate_p75_s"], C_GPU, "Surrogate (GPU)")
axL.set_xscale("log"); axL.set_yscale("log")
axL.set_xlabel("Tamaño de lote", fontsize=10.5)
axL.set_ylabel("Tiempo (s)", fontsize=10.5)
axL.set_title("Tiempo de evaluación", fontsize=11)
axL.legend(fontsize=8.5, loc="upper left")

# --- Panel derecho: speedup vs lote ------------------------------------------
axR.plot(batches, cpu["speedup_median"], "o-", color=C_CPU, lw=1.8, ms=6, label="CPU",
         markeredgecolor=EDGE, markeredgewidth=0.6)
axR.plot(batches, gpu["speedup_median"], "o-", color=C_GPU, lw=1.8, ms=6, label="GPU",
         markeredgecolor=EDGE, markeredgewidth=0.6)
axR.set_xscale("log"); axR.set_yscale("log")
axR.set_xlabel("Tamaño de lote", fontsize=10.5)
axR.set_ylabel(r"Speedup ($t_{\mathrm{solver}}/t_{\mathrm{surrogate}}$)", fontsize=10.5)
axR.set_title("Aceleración frente al solver", fontsize=11)
axR.legend(fontsize=8.5, loc="upper left")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
