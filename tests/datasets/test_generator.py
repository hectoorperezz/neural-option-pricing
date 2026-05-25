import numpy as np
import pytest

from src.datasets import DatasetGenerator, UniformSampler, make_black_scholes_domain
from src.solvers import BlackScholesSolver


def test_black_scholes_generator_returns_normalized_features_price_and_delta() -> None:
    domain = make_black_scholes_domain()
    solver = BlackScholesSolver()
    sampler = UniformSampler(domain, seed=123)
    generator = DatasetGenerator(
        solver=solver,
        domain=domain,
        sampler=sampler,
        model_family="black_scholes",
        include_delta=True,
    )

    dataset = generator.generate(16, seed=321)
    sample = dataset[0]
    raw = sample["raw_inputs"]

    reference_price = solver.call_price(raw[0].item(), 1.0, raw[1].item(), raw[2].item(), raw[3].item())
    reference_delta = solver.call_delta(raw[0].item(), 1.0, raw[1].item(), raw[2].item(), raw[3].item())

    assert len(dataset) == 16
    assert dataset.features.shape == (16, domain.dimension)
    assert dataset.prices.shape == (16, 1)
    assert dataset.deltas is not None
    assert dataset.deltas.shape == (16, 1)
    assert float(dataset.features.min()) >= 0.0
    assert float(dataset.features.max()) <= 1.0
    assert sample["price"].item() == pytest.approx(reference_price, abs=1e-6)
    assert sample["delta"].item() == pytest.approx(reference_delta, abs=1e-6)


def test_generate_batch_returns_validity_counters_and_numpy_arrays() -> None:
    domain = make_black_scholes_domain()
    generator = DatasetGenerator(
        solver=BlackScholesSolver(),
        domain=domain,
        sampler=UniformSampler(domain),
        model_family="black_scholes",
        include_delta=False,
    )

    batch = generator.generate_batch(32, rng=np.random.default_rng(123))

    assert batch.attempted_count == 32
    assert batch.accepted_count + batch.rejected_count == 32
    assert batch.features.shape == (batch.accepted_count, domain.dimension)
    assert batch.raw_inputs.shape == (batch.accepted_count, domain.dimension)
    assert batch.prices.shape == (batch.accepted_count,)
    assert batch.deltas is None


def test_generator_rejects_unknown_model_family() -> None:
    domain = make_black_scholes_domain()
    generator = DatasetGenerator(
        solver=BlackScholesSolver(),
        domain=domain,
        sampler=UniformSampler(domain),
        model_family="unknown",
    )

    with pytest.raises(ValueError, match="model_family"):
        generator.generate(1)
