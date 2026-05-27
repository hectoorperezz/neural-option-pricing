"""Bin partition for the 5x5 moneyness x maturity evaluation grid.

Defines :class:`BinPartition`, the canonical way to assign each test point
to one of the 25 bins used by E1-E5 to report per-region surrogate quality.
The bin boundaries are imported from :mod:`src.datasets.sampler` so that
they stay aligned with the boundaries used to draw the balanced test set.

Binning convention: ``[low, high)`` for all bins except the topmost one on
each axis, which is ``[low, high]``. The documentation also lists ATM as
``[0.9, 1.1]`` (closed on both sides), but the uniform sampler used to
generate the test set never lands exactly on a boundary, so adopting the
plain right-open convention everywhere keeps the implementation simpler
without changing the assignment of any actually-sampled point.

The ``bin_id`` returned by :meth:`BinPartition.assign` follows the same
layout used by :class:`src.datasets.sampler.BalancedBinSampler.iter_bins`::

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
    """Partition of the (moneyness, maturity) plane into a regular grid.

    Each axis is sliced into a sequence of half-open intervals ``[low, high)``,
    with the rightmost interval closed on both ends. A test point is mapped
    to a single bin by locating the interval that contains it on each axis
    and combining the two indices.
    """

    moneyness_bins: tuple[tuple[float, float], ...]
    maturity_bins: tuple[tuple[float, float], ...]
    moneyness_labels: tuple[str, ...] = _DEFAULT_MONEYNESS_LABELS
    maturity_labels: tuple[str, ...] = _DEFAULT_MATURITY_LABELS

    @classmethod
    def default(cls) -> "BinPartition":
        """Standard 5x5 partition shared with :class:`BalancedBinSampler`."""
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
        """Assign each (moneyness, maturity) pair to its bin.

        Returns a tuple ``(bin_id, moneyness_idx, maturity_idx)`` of integer
        arrays sharing the broadcast shape of the inputs. Out-of-domain
        inputs raise :class:`ValueError`; we never silently clamp.
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
        """Human-readable label for a bin, e.g. ``"atm_weekly"``."""
        return (
            f"{self.moneyness_labels[moneyness_idx]}"
            f"_{self.maturity_labels[maturity_idx]}"
        )

    def bin_id_from_indices(self, moneyness_idx: int, maturity_idx: int) -> int:
        """Inverse of the bin layout: indices → ``bin_id``."""
        if not 0 <= moneyness_idx < self.n_moneyness_bins:
            raise ValueError(
                f"moneyness_idx must be in [0, {self.n_moneyness_bins})"
            )
        if not 0 <= maturity_idx < self.n_maturity_bins:
            raise ValueError(
                f"maturity_idx must be in [0, {self.n_maturity_bins})"
            )
        return maturity_idx * self.n_moneyness_bins + moneyness_idx
