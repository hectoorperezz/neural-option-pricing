"""Partición 5x5 de moneyness y vencimiento para evaluación.

``BinPartition`` asigna cada punto de test a uno de los 25 bins usados en
E1-E5. Los límites se importan desde ``src.datasets.sampler`` para que la
evaluación y el test set balanced usen exactamente la misma rejilla.

La convención es ``[low, high)`` salvo el último bin de cada eje, que es
``[low, high]``. Así los puntos de frontera se asignan de forma
determinista.

El ``bin_id`` sigue el mismo orden que ``BalancedBinSampler.iter_bins``::

    bin_id = maturity_idx * n_moneyness_bins + moneyness_idx
"""


from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.datasets.sampler import MATURITY_BINS, MONEYNESS_BINS


_DEFAULT_MONEYNESS_LABELS: tuple[str, ...] = (
    "deep_otm",
    "otm",
    "atm",
    "itm",
    "deep_itm",
)
_DEFAULT_MATURITY_LABELS: tuple[str, ...] = (
    "weekly",
    "short",
    "medium_short",
    "medium",
    "long",
)


@dataclass(frozen=True)
class BinPartition:
    """Rejilla regular sobre el plano ``(moneyness, maturity)``.

    Cada eje se divide en intervalos. Un punto se asigna localizando su
    intervalo de moneyness y su intervalo de vencimiento, y combinando ambos
    índices.
    """

    moneyness_bins: tuple[tuple[float, float], ...]
    maturity_bins: tuple[tuple[float, float], ...]
    moneyness_labels: tuple[str, ...] = _DEFAULT_MONEYNESS_LABELS
    maturity_labels: tuple[str, ...] = _DEFAULT_MATURITY_LABELS

    @classmethod
    def default(cls) -> "BinPartition":
        """Partición 5x5 estándar compartida con ``BalancedBinSampler``."""
        return cls(moneyness_bins=MONEYNESS_BINS, maturity_bins=MATURITY_BINS)

    def __post_init__(self) -> None:
        if len(self.moneyness_bins) == 0:
            raise ValueError("moneyness_bins must contain at least one bin")
        if len(self.maturity_bins) == 0:
            raise ValueError("maturity_bins must contain at least one bin")
        if len(self.moneyness_labels) != len(self.moneyness_bins):
            raise ValueError(
                "moneyness_labels must have one entry per moneyness bin "
                f"(got {len(self.moneyness_labels)} labels, "
                f"{len(self.moneyness_bins)} bins)"
            )
        if len(self.maturity_labels) != len(self.maturity_bins):
            raise ValueError(
                "maturity_labels must have one entry per maturity bin "
                f"(got {len(self.maturity_labels)} labels, "
                f"{len(self.maturity_bins)} bins)"
            )

    @property
    def n_moneyness_bins(self) -> int:
        return len(self.moneyness_bins)

    @property
    def n_maturity_bins(self) -> int:
        return len(self.maturity_bins)

    @property
    def n_bins(self) -> int:
        return self.n_moneyness_bins * self.n_maturity_bins

    @property
    def moneyness_range(self) -> tuple[float, float]:
        return self.moneyness_bins[0][0], self.moneyness_bins[-1][1]

    @property
    def maturity_range(self) -> tuple[float, float]:
        return self.maturity_bins[0][0], self.maturity_bins[-1][1]

    def assign(
        self,
        moneyness: np.ndarray | float,
        maturity: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Asigna cada par ``(moneyness, maturity)`` a su bin.

        Devuelve ``(bin_id, moneyness_idx, maturity_idx)``. Si un punto cae
        fuera del dominio, se lanza ``ValueError``; no se recorta en silencio.
        """
        moneyness_arr = np.asarray(moneyness, dtype=np.float64)
        maturity_arr = np.asarray(maturity, dtype=np.float64)

        if moneyness_arr.shape != maturity_arr.shape:
            raise ValueError(
                "moneyness and maturity must have the same shape "
                f"(got {moneyness_arr.shape} and {maturity_arr.shape})"
            )

        m_lo, m_hi = self.moneyness_range
        t_lo, t_hi = self.maturity_range
        if np.any(moneyness_arr < m_lo) or np.any(moneyness_arr > m_hi):
            raise ValueError(
                f"moneyness values must lie inside [{m_lo}, {m_hi}]"
            )
        if np.any(maturity_arr < t_lo) or np.any(maturity_arr > t_hi):
            raise ValueError(
                f"maturity values must lie inside [{t_lo}, {t_hi}]"
            )

        moneyness_edges = np.array(
            [bin_bounds[0] for bin_bounds in self.moneyness_bins]
            + [self.moneyness_bins[-1][1]],
            dtype=np.float64,
        )
        maturity_edges = np.array(
            [bin_bounds[0] for bin_bounds in self.maturity_bins]
            + [self.maturity_bins[-1][1]],
            dtype=np.float64,
        )

        m_idx = np.searchsorted(moneyness_edges, moneyness_arr, side="right") - 1
        t_idx = np.searchsorted(maturity_edges, maturity_arr, side="right") - 1
        m_idx = np.clip(m_idx, 0, self.n_moneyness_bins - 1)
        t_idx = np.clip(t_idx, 0, self.n_maturity_bins - 1)

        bin_id = t_idx * self.n_moneyness_bins + m_idx
        return (
            bin_id.astype(np.int64, copy=False),
            m_idx.astype(np.int64, copy=False),
            t_idx.astype(np.int64, copy=False),
        )

    def bin_label(self, moneyness_idx: int, maturity_idx: int) -> str:
        """Etiqueta legible del bin, por ejemplo ``"atm_weekly"``."""
        return (
            f"{self.moneyness_labels[moneyness_idx]}"
            f"_{self.maturity_labels[maturity_idx]}"
        )

    def bin_id_from_indices(self, moneyness_idx: int, maturity_idx: int) -> int:
        """Conversión inversa de índices de rejilla a ``bin_id``."""
        if not 0 <= moneyness_idx < self.n_moneyness_bins:
            raise ValueError(
                f"moneyness_idx must be in [0, {self.n_moneyness_bins})"
            )
        if not 0 <= maturity_idx < self.n_maturity_bins:
            raise ValueError(
                f"maturity_idx must be in [0, {self.n_maturity_bins})"
            )
        return maturity_idx * self.n_moneyness_bins + moneyness_idx
