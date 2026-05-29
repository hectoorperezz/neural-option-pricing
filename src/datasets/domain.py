"""Dominios de entrada y normalización min-max.

Define el hipercubo de parámetros de cada modelo (Black-Scholes y
Heston) y centraliza la normalización a ``[0, 1]`` que ven las redes.
Los nombres declarados aquí son los que viajan en los ``.npz`` y los
que el resto del proyecto usa como contrato (ver
``docs/metodologia.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Domain:
    """Hipercubo de entrada y transformaciones min-max deterministas.

    Attributes:
        input_names: Nombres en el mismo orden que ``lower_bounds`` y
            ``upper_bounds``. Define la columna de cada feature.
        lower_bounds: Cota inferior por dimensión.
        upper_bounds: Cota superior por dimensión.
        sqrt_sampled_names: Dimensiones donde el muestreo uniforme se
            hace en ``√x`` y luego se eleva al cuadrado, para
            concentrar masa cerca de cero (varianzas en Heston).
        dividend_yield: ``q`` global del dominio; por convención del
            proyecto vale ``0`` (ver E5 y la equivalencia Delta = P_1).
    """

    input_names: tuple[str, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    sqrt_sampled_names: tuple[str, ...] = ()
    dividend_yield: float = 0.0

    @property
    def dimension(self) -> int:
        """Número de features del dominio."""
        return len(self.input_names)

    @property
    def lower(self) -> np.ndarray:
        return np.asarray(self.lower_bounds, dtype=float)

    @property
    def upper(self) -> np.ndarray:
        return np.asarray(self.upper_bounds, dtype=float)

    def normalize(self, raw_inputs: np.ndarray) -> np.ndarray:
        """Lleva ``raw_inputs`` al cubo ``[0, 1]^d`` por min-max."""
        values = np.asarray(raw_inputs, dtype=float)
        return (values - self.lower) / (self.upper - self.lower)

    def denormalize(self, normalized_inputs: np.ndarray) -> np.ndarray:
        """Inversa de :meth:`normalize`; útil para evaluar el solver."""
        values = np.asarray(normalized_inputs, dtype=float)
        return self.lower + values * (self.upper - self.lower)

    def sample_uniform(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Muestrea ``n_samples`` puntos del hipercubo.

        Las dimensiones listadas en ``sqrt_sampled_names`` se muestrean
        en raíz cuadrada y se elevan después, replicando el sampler
        descrito en ``docs/metodologia.md`` para las varianzas de Heston.
        """
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
        """Devuelve ``2κθ - ξ²`` por fila; positivo ⇒ se cumple Feller."""
        values = np.asarray(raw_inputs, dtype=float)
        kappa = values[:, self.input_names.index("kappa")]
        theta = values[:, self.input_names.index("theta")]
        xi = values[:, self.input_names.index("xi")]
        return 2.0 * kappa * theta - xi * xi


def make_black_scholes_domain() -> Domain:
    """Dominio de los surrogates BS-1..BS-4."""
    return Domain(
        input_names=("moneyness", "maturity", "rate", "volatility"),
        lower_bounds=(0.4, 7.0 / 365.0, 0.0, 0.03),
        upper_bounds=(2.0, 2.0, 0.075, 1.0),
    )


def make_heston_domain() -> Domain:
    """Dominio de los surrogates H-1..H-6, con varianzas en ``√``."""
    return Domain(
        input_names=("moneyness", "maturity", "rate", "v0", "theta", "kappa", "xi", "rho"),
        lower_bounds=(0.4, 7.0 / 365.0, 0.0, 0.0009, 0.0009, 0.10, 0.10, -0.95),
        upper_bounds=(2.0, 2.0, 0.075, 1.0, 1.0, 10.0, 3.0, -0.05),
        sqrt_sampled_names=("v0", "theta"),
    )
