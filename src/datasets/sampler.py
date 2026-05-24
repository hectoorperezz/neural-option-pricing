from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.datasets.domain import Domain


@dataclass(frozen=True)
class UniformSampler:
    """Uniform sampler in the financial scale defined by the domain."""

    domain: Domain
    seed: int | None = None

    def sample(self, n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = np.random.default_rng(self.seed) if rng is None else rng
        return self.domain.sample_uniform(n_samples, rng)


@dataclass(frozen=True)
class FocusedSampler:
    """Mixture sampler for H-5: global coverage plus ATM/short-maturity focus."""

    domain: Domain
    seed: int | None = None
    focused_probability: float = 0.5
    moneyness_mean: float = 1.0
    moneyness_std: float = 0.15
    moneyness_bounds: tuple[float, float] = (0.7, 1.3)
    maturity_bounds: tuple[float, float] = (7.0 / 365.0, 0.25)

    def sample(self, n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
        if not 0.0 <= self.focused_probability <= 1.0:
            raise ValueError("focused_probability must be between 0 and 1")

        rng = np.random.default_rng(self.seed) if rng is None else rng
        samples = self.domain.sample_uniform(n_samples, rng)
        focused_mask = rng.random(n_samples) < self.focused_probability
        focused_count = int(focused_mask.sum())
        if focused_count == 0:
            return samples

        moneyness_index = self.domain.input_names.index("moneyness")
        maturity_index = self.domain.input_names.index("maturity")
        samples[focused_mask, moneyness_index] = self._sample_truncated_normal(
            focused_count,
            rng,
            self.moneyness_mean,
            self.moneyness_std,
            self.moneyness_bounds[0],
            self.moneyness_bounds[1],
        )
        samples[focused_mask, maturity_index] = self._sample_log_uniform(
            focused_count,
            rng,
            self.maturity_bounds[0],
            self.maturity_bounds[1],
        )
        return samples

    def _sample_truncated_normal(
        self,
        n_samples: int,
        rng: np.random.Generator,
        mean: float,
        std: float,
        lower: float,
        upper: float,
    ) -> np.ndarray:
        values = np.empty(n_samples, dtype=float)
        filled = 0
        while filled < n_samples:
            draw_count = max(2 * (n_samples - filled), 32)
            draws = rng.normal(mean, std, size=draw_count)
            accepted = draws[(lower <= draws) & (draws <= upper)]
            take = min(n_samples - filled, accepted.size)
            values[filled : filled + take] = accepted[:take]
            filled += take
        return values

    def _sample_log_uniform(
        self,
        n_samples: int,
        rng: np.random.Generator,
        lower: float,
        upper: float,
    ) -> np.ndarray:
        if lower <= 0.0 or upper <= lower:
            raise ValueError("log-uniform bounds must satisfy 0 < lower < upper")
        return np.exp(rng.uniform(np.log(lower), np.log(upper), size=n_samples))
