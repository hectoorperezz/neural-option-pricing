import numpy as np
import pytest

from src.datasets.sampler import MATURITY_BINS, MONEYNESS_BINS
from src.evaluation import BinPartition


def test_default_partition_has_25_bins() -> None:
    partition = BinPartition.default()

    assert partition.n_moneyness_bins == 5
    assert partition.n_maturity_bins == 5
    assert partition.n_bins == 25


def test_default_partition_reuses_sampler_constants() -> None:
    partition = BinPartition.default()

    assert partition.moneyness_bins == MONEYNESS_BINS
    assert partition.maturity_bins == MATURITY_BINS


def test_assign_scalar_in_deep_otm_medium_short() -> None:
    partition = BinPartition.default()

    bin_id, m_idx, t_idx = partition.assign(0.5, 0.1)

    assert int(m_idx) == 0  # deep_otm
    assert int(t_idx) == 2  # medium_short
    assert int(bin_id) == 2 * 5 + 0


def test_assign_scalar_at_atm_medium() -> None:
    partition = BinPartition.default()

    bin_id, m_idx, t_idx = partition.assign(1.0, 0.5)

    assert int(m_idx) == 2  # atm
    assert int(t_idx) == 3  # medium
    assert int(bin_id) == 3 * 5 + 2


def test_assign_scalar_at_deep_itm_long() -> None:
    partition = BinPartition.default()

    bin_id, m_idx, t_idx = partition.assign(1.5, 1.5)

    assert int(m_idx) == 4  # deep_itm
    assert int(t_idx) == 4  # long
    assert int(bin_id) == 4 * 5 + 4


def test_assign_vectorised_returns_same_shape() -> None:
    partition = BinPartition.default()
    moneyness = np.array([0.5, 1.0, 1.5])
    maturity = np.array([0.1, 0.5, 1.5])

    bin_id, m_idx, t_idx = partition.assign(moneyness, maturity)

    assert bin_id.shape == (3,)
    assert m_idx.tolist() == [0, 2, 4]
    assert t_idx.tolist() == [2, 3, 4]


def test_assign_left_boundary_is_inclusive() -> None:
    partition = BinPartition.default()

    # m = 0.9 is the lower edge of ATM ([0.9, 1.1)); it belongs to ATM.
    _, m_idx, _ = partition.assign(0.9, 0.5)

    assert int(m_idx) == 2


def test_assign_internal_right_boundary_moves_to_next_bin() -> None:
    partition = BinPartition.default()

    _, m_idx_11, _ = partition.assign(1.1, 0.5)
    _, m_idx_13, _ = partition.assign(1.3, 0.5)

    assert int(m_idx_11) == 3  # ITM
    assert int(m_idx_13) == 4  # deep_itm


def test_assign_upper_extreme_falls_in_last_bin() -> None:
    partition = BinPartition.default()

    bin_id, m_idx, t_idx = partition.assign(2.0, 2.0)

    assert int(m_idx) == 4
    assert int(t_idx) == 4
    assert int(bin_id) == 24


def test_assign_lower_extreme_falls_in_first_bin() -> None:
    partition = BinPartition.default()

    _, m_idx, t_idx = partition.assign(0.4, 7.0 / 365.0)

    assert int(m_idx) == 0
    assert int(t_idx) == 0


def test_assign_raises_for_out_of_domain_moneyness() -> None:
    partition = BinPartition.default()

    with pytest.raises(ValueError, match="moneyness"):
        partition.assign(0.3, 0.5)

    with pytest.raises(ValueError, match="moneyness"):
        partition.assign(2.5, 0.5)


def test_assign_raises_for_out_of_domain_maturity() -> None:
    partition = BinPartition.default()

    with pytest.raises(ValueError, match="maturity"):
        partition.assign(1.0, 0.001)

    with pytest.raises(ValueError, match="maturity"):
        partition.assign(1.0, 3.0)


def test_assign_raises_on_shape_mismatch() -> None:
    partition = BinPartition.default()

    with pytest.raises(ValueError, match="shape"):
        partition.assign(np.array([1.0, 1.1]), np.array([0.5]))


def test_bin_label_atm_weekly() -> None:
    partition = BinPartition.default()

    assert partition.bin_label(2, 0) == "atm_weekly"


def test_bin_label_deep_itm_long() -> None:
    partition = BinPartition.default()

    assert partition.bin_label(4, 4) == "deep_itm_long"


def test_bin_id_layout_matches_balanced_sampler() -> None:
    """``bin_id`` must obey the same formula as ``BalancedBinSampler.iter_bins``."""
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


def test_assign_array_returns_correct_shape_and_range() -> None:
    partition = BinPartition.default()
    rng = np.random.default_rng(42)
    moneyness = rng.uniform(0.4, 2.0, size=1000)
    maturity = rng.uniform(7.0 / 365.0, 2.0, size=1000)

    bin_id, m_idx, t_idx = partition.assign(moneyness, maturity)

    assert bin_id.shape == (1000,)
    assert m_idx.shape == (1000,)
    assert t_idx.shape == (1000,)
    assert np.all((bin_id >= 0) & (bin_id < 25))
    assert np.all((m_idx >= 0) & (m_idx < 5))
    assert np.all((t_idx >= 0) & (t_idx < 5))


def test_custom_partition_with_different_dimensions() -> None:
    partition = BinPartition(
        moneyness_bins=((0.5, 1.0), (1.0, 1.5)),
        maturity_bins=((0.1, 0.5), (0.5, 1.0), (1.0, 2.0)),
        moneyness_labels=("low", "high"),
        maturity_labels=("short", "medium", "long"),
    )

    assert partition.n_bins == 6
    assert partition.n_moneyness_bins == 2
    assert partition.n_maturity_bins == 3

    bin_id, m_idx, t_idx = partition.assign(0.7, 1.5)
    assert int(m_idx) == 0
    assert int(t_idx) == 2
    assert int(bin_id) == 2 * 2 + 0
    assert partition.bin_label(int(m_idx), int(t_idx)) == "low_long"


def test_label_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="moneyness_labels"):
        BinPartition(
            moneyness_bins=((0.4, 1.0), (1.0, 2.0)),
            maturity_bins=((0.1, 1.0),),
            moneyness_labels=("only_one",),
            maturity_labels=("only_one",),
        )


def test_bin_id_from_indices_round_trip() -> None:
    partition = BinPartition.default()

    for t_idx in range(5):
        for m_idx in range(5):
            bin_id = partition.bin_id_from_indices(m_idx, t_idx)
            assert bin_id == t_idx * 5 + m_idx


def test_bin_id_from_indices_rejects_out_of_range() -> None:
    partition = BinPartition.default()

    with pytest.raises(ValueError, match="moneyness_idx"):
        partition.bin_id_from_indices(5, 0)
    with pytest.raises(ValueError, match="maturity_idx"):
        partition.bin_id_from_indices(0, 5)
