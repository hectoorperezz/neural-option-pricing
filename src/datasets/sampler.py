from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.datasets.domain import Domain


MONEYNESS_BINS: tuple[tuple[float, float], ...] = (
    (0.4, 0.7),
    (0.7, 0.9),
    (0.9, 1.1),
    (1.1, 1.3),
    (1.3, 2.0),
)
MATURITY_BINS: tuple[tuple[float, float], ...] = (
    (7.0 / 365.0, 14.0 / 365.0),
    (14.0 / 365.0, 1.0 / 12.0),
    (1.0 / 12.0, 0.25),
    (0.25, 1.0),
    (1.0, 2.0),
)


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


@dataclass(frozen=True)
class BalancedBinSampler:
    """Sampler with equal observations in each moneyness x maturity bin."""

    domain: Domain
    samples_per_bin: int
    seed: int | None = None
    moneyness_bins: tuple[tuple[float, float], ...] = MONEYNESS_BINS
    maturity_bins: tuple[tuple[float, float], ...] = MATURITY_BINS

    @property
    def n_bins(self) -> int:
        return len(self.moneyness_bins) * len(self.maturity_bins)

    @property
    def total_samples(self) -> int:
        return self.samples_per_bin * self.n_bins

    def iter_bins(self):
        for maturity_index, maturity_bounds in enumerate(self.maturity_bins):
            for moneyness_index, moneyness_bounds in enumerate(self.moneyness_bins):
                bin_id = maturity_index * len(self.moneyness_bins) + moneyness_index
                yield bin_id, moneyness_index, maturity_index, moneyness_bounds, maturity_bounds

    def sample(self, n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
        if n_samples != self.total_samples:
            raise ValueError("n_samples must equal samples_per_bin * number_of_bins")

        rng = np.random.default_rng(self.seed) if rng is None else rng
        parts = [
            self.sample_bin(moneyness_bounds, maturity_bounds, self.samples_per_bin, rng)
            for _, _, _, moneyness_bounds, maturity_bounds in self.iter_bins()
        ]
        samples = np.concatenate(parts, axis=0)
        rng.shuffle(samples)
        return samples

    def sample_bin(
        self,
        moneyness_bounds: tuple[float, float],
        maturity_bounds: tuple[float, float],
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if self.samples_per_bin <= 0:
            raise ValueError("samples_per_bin must be strictly positive")
        if n_samples <= 0:
            raise ValueError("n_samples must be strictly positive")

        samples = self.domain.sample_uniform(n_samples, rng)
        moneyness_index = self.domain.input_names.index("moneyness")
        maturity_index = self.domain.input_names.index("maturity")
        samples[:, moneyness_index] = rng.uniform(
            moneyness_bounds[0], moneyness_bounds[1], size=n_samples
        )
        samples[:, maturity_index] = rng.uniform(
            maturity_bounds[0], maturity_bounds[1], size=n_samples
        )
        return samples
