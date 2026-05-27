from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Domain:
    """Input hypercube and deterministic min-max transforms."""

    input_names: tuple[str, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    sqrt_sampled_names: tuple[str, ...] = ()
    dividend_yield: float = 0.0

    @property
    def dimension(self) -> int:
        return len(self.input_names)

    @property
    def lower(self) -> np.ndarray:
        return np.asarray(self.lower_bounds, dtype=float)

    @property
    def upper(self) -> np.ndarray:
        return np.asarray(self.upper_bounds, dtype=float)

    def normalize(self, raw_inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_inputs, dtype=float)
        return (values - self.lower) / (self.upper - self.lower)

    def denormalize(self, normalized_inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(normalized_inputs, dtype=float)
        return self.lower + values * (self.upper - self.lower)

    def sample_uniform(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative")

        samples = rng.uniform(self.lower, self.upper, size=(n_samples, self.dimension))
        for name in self.sqrt_sampled_names:
            index = self.input_names.index(name)
            low = np.sqrt(self.lower[index])
            high = np.sqrt(self.upper[index])
            samples[:, index] = rng.uniform(low, high, size=n_samples) ** 2
        return samples

    def feller_values(self, raw_inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_inputs, dtype=float)
        kappa = values[:, self.input_names.index("kappa")]
        theta = values[:, self.input_names.index("theta")]
        xi = values[:, self.input_names.index("xi")]
        return 2.0 * kappa * theta - xi * xi


def make_black_scholes_domain() -> Domain:
    return Domain(
        input_names=("moneyness", "maturity", "rate", "volatility"),
        lower_bounds=(0.4, 7.0 / 365.0, 0.0, 0.03),
        upper_bounds=(2.0, 2.0, 0.075, 1.0),
    )


def make_heston_domain() -> Domain:
    return Domain(
        input_names=("moneyness", "maturity", "rate", "v0", "theta", "kappa", "xi", "rho"),
        lower_bounds=(0.4, 7.0 / 365.0, 0.0, 0.0009, 0.0009, 0.10, 0.10, -0.95),
        upper_bounds=(2.0, 2.0, 0.075, 1.0, 1.0, 10.0, 3.0, -0.05),
        sqrt_sampled_names=("v0", "theta"),
    )
