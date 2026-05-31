"""Identidad visual común para las figuras del paper.

Centraliza la paleta institucional (amarillo Hespérides / negro) y un conjunto
de ``rcParams`` de matplotlib para que todos los plots generados por script
compartan el mismo estilo que el documento LaTeX: tinta casi negra, tipografía
serif, rejilla tenue y acentos en amarillo.

Uso:
    from paper_style import apply_paper_style, PALETTE, ACT_COLOR, heatmap_cmap
    apply_paper_style()
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.colors as mcolors

# --- Paleta institucional (coincide con paper.tex) --------------------------
PALETTE = {
    "uniyellow": "#FFD100",
    "uniyellowsoft": "#FDEA99",
    "uniblack": "#1A1A1A",
    "grey": "#3A3A3A",
    "gold": "#8A6D00",
    "red": "#B83A3A",
    "orange": "#D96B1C",
    "green": "#3E8E41",
    "purple": "#6F3FAE",
}

# Colores por activación, iguales a la figura de activaciones del paper.
ACT_COLOR = {
    "relu": PALETTE["red"],
    "softplus": PALETTE["orange"],
    "swish": PALETTE["green"],
    "tanh": PALETTE["purple"],
}
ACT_LABEL = {"relu": "ReLU", "softplus": "Softplus", "swish": "Swish", "tanh": "tanh"}


# Etiquetas de bins en terminología técnica (inglés), igual que fig:bins.
MONEYNESS_LABELS = ("Deep\nOTM", "OTM", "ATM", "ITM", "Deep\nITM")
MATURITY_LABELS = ("Weekly", "Short", "Medium-\nshort", "Medium", "Long")


def heatmap_cmap() -> mcolors.Colormap:
    """Mapa secuencial blanco -> amarillo -> negro para heatmaps."""
    return mcolors.LinearSegmentedColormap.from_list(
        "hesperides", ["#FFFDF2", PALETTE["uniyellow"], "#8A6D00", PALETTE["uniblack"]]
    )


def diverging_cmap() -> mcolors.Colormap:
    """Mapa divergente rojo (peor) -> crema (igual) -> verde (mejor)."""
    return mcolors.LinearSegmentedColormap.from_list(
        "e3div", [PALETTE["red"], "#FBF3D6", PALETTE["green"]]
    )


def apply_paper_style() -> None:
    """Fija los ``rcParams`` de la identidad visual del documento."""
    ink = PALETTE["uniblack"]
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "text.color": ink,
        "axes.edgecolor": ink,
        "axes.labelcolor": ink,
        "axes.titlecolor": ink,
        "axes.linewidth": 0.9,
        "xtick.color": ink,
        "ytick.color": ink,
        "axes.grid": True,
        "grid.color": "#D9D2BE",      # gris cálido tirando a la crema del paper
        "grid.linewidth": 0.5,
        "grid.linestyle": ":",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def accent_title(ax, text: str, fontsize: float = 12.0) -> None:
    """Título con un subrayado amarillo, como los acentos del documento."""
    ax.set_title(text, fontsize=fontsize, fontweight="bold", pad=10)
