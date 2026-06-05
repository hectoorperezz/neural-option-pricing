"""Figura ``fig:samplers`` del paper: muestreo uniforme frente a enfocado.

Genera la comparación en el plano ``(m, T)`` usando los samplers reales del
repositorio (``UniformSampler`` y ``FocusedSampler``) sobre el dominio de
Heston, con la identidad visual del documento. Escribe el PNG en
``results/figures/`` y en ``docs/latex/assets/`` para que el paper lo incluya.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # paper_style (hermano)

from paper_style import PALETTE, apply_paper_style  # noqa: E402
from src.datasets import (  # noqa: E402
    FocusedSampler,
    UniformSampler,
    make_heston_domain,
)


def main() -> None:
    apply_paper_style()
    domain = make_heston_domain()
    mi = domain.input_names.index("moneyness")
    ti = domain.input_names.index("maturity")

    rng = np.random.default_rng(0)
    unif = UniformSampler(domain).sample(4000, rng=rng)
    foc = FocusedSampler(domain).sample(4000, rng=rng)

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.7), sharex=True, sharey=True)
    ax[0].scatter(unif[:, mi], unif[:, ti], s=4, color=PALETTE["uniblack"], alpha=0.30)
    ax[0].set_title("Muestreo uniforme", fontsize=12, fontweight="bold", pad=8)
    ax[1].scatter(foc[:, mi], foc[:, ti], s=4, color=PALETTE["gold"], alpha=0.30)
    ax[1].set_title("Muestreo enfocado", fontsize=12, fontweight="bold", pad=8)
    ax[0].set_ylabel("$T$ (años)")
    for a in ax:
        a.set_xlabel(r"$m = S/K$")
        a.set_xlim(0.4, 2.0)
        a.set_ylim(0.0, 2.0)
    fig.tight_layout()

    for out_dir in (REPO_ROOT / "results" / "figures", REPO_ROOT / "docs" / "latex" / "assets"):
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "samplers.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print("escrito", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
