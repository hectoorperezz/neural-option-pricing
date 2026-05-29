"""Test del contrato entre ``BinPartition`` y ``BalancedBinSampler``.

Dos componentes del proyecto numeran los 25 bins de la rejilla 5x5:
el sampler los etiqueta al generar el test set y la partición los
reasigna al evaluar el surrogate. Si las dos fórmulas difirieran, los
``MAE`` se reportarían en el bin equivocado y todos los heatmaps
serían incorrectos sin que ningún E2E lo detectara (los CSV y los
PNG se siguen produciendo, solo que con contenido erróneo). Este test
fija el contrato.
"""

from src.datasets.sampler import MATURITY_BINS, MONEYNESS_BINS
from src.evaluation import BinPartition


def test_bin_id_layout_matches_balanced_sampler() -> None:
    """``bin_id`` sigue la misma fórmula que ``BalancedBinSampler.iter_bins``."""
    partition = BinPartition.default()

    for t_idx, (t_lo, t_hi) in enumerate(MATURITY_BINS):
        for m_idx, (m_lo, m_hi) in enumerate(MONEYNESS_BINS):
            expected_bin_id = t_idx * len(MONEYNESS_BINS) + m_idx
            m_center = 0.5 * (m_lo + m_hi)
            t_center = 0.5 * (t_lo + t_hi)

            bin_id, m_assigned, t_assigned = partition.assign(m_center, t_center)

            assert int(bin_id) == expected_bin_id
            assert int(m_assigned) == m_idx
            assert int(t_assigned) == t_idx
