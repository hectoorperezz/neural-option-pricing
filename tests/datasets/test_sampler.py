"""Tests de los samplers.

Verifican tres propiedades estadísticas que un E2E del pipeline no
detectaría aunque estuvieran rotas:

* el sesgo en ``√varianza`` para Heston (documentado en
  ``docs/metodologia.md``);
* la concentración ATM y corto plazo del ``FocusedSampler``, sin la
  que H-3 y H-5 serían indistinguibles en E3;
* el conteo exacto por bin del ``BalancedBinSampler``, necesario para
  que los ``MAE`` por bin sean comparables.
"""

from collections import Counter

import numpy as np

from src.datasets import (
    BalancedBinSampler,
    FocusedSampler,
    UniformSampler,
    make_heston_domain,
)


def test_heston_sampler_samples_variances_in_sqrt_scale() -> None:
    """Para ``v0``, la media de ``√v0`` se sitúa cerca del centro del rango raíz."""
    domain = make_heston_domain()
    samples = UniformSampler(domain, seed=123).sample(2_000)
    v0 = samples[:, domain.input_names.index("v0")]

    assert np.all(v0 >= 0.0009)
    assert np.all(v0 <= 1.0)
    assert 0.48 < np.mean(np.sqrt(v0)) < 0.55


def test_focused_sampler_concentrates_moneyness_and_maturity() -> None:
    """El sampler enfocado aumenta la masa ATM y reduce la mediana de plazo."""
    domain = make_heston_domain()
    uniform = UniformSampler(domain, seed=7).sample(5_000)
    focused = FocusedSampler(domain, seed=7).sample(5_000)
    m_index = domain.input_names.index("moneyness")
    t_index = domain.input_names.index("maturity")

    uniform_atm_share = np.mean((0.7 <= uniform[:, m_index]) & (uniform[:, m_index] <= 1.3))
    focused_atm_share = np.mean((0.7 <= focused[:, m_index]) & (focused[:, m_index] <= 1.3))

    assert focused_atm_share > uniform_atm_share
    assert np.median(focused[:, t_index]) < np.median(uniform[:, t_index])


def test_balanced_bin_sampler_generates_equal_counts_per_bin() -> None:
    """Exactamente ``samples_per_bin`` puntos en cada uno de los 25 bins."""
    domain = make_heston_domain()
    sampler = BalancedBinSampler(domain, samples_per_bin=3, seed=123)
    samples = sampler.sample(75)
    m_index = domain.input_names.index("moneyness")
    t_index = domain.input_names.index("maturity")
    counts: Counter[tuple[int, int]] = Counter()

    for sample in samples:
        m_bin = next(
            index
            for index, bounds in enumerate(sampler.moneyness_bins)
            if bounds[0] <= sample[m_index] <= bounds[1]
        )
        t_bin = next(
            index
            for index, bounds in enumerate(sampler.maturity_bins)
            if bounds[0] <= sample[t_index] <= bounds[1]
        )
        counts[(m_bin, t_bin)] += 1

    assert len(counts) == 25
    assert set(counts.values()) == {3}
