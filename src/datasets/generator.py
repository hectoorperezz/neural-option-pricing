"""Generación de datasets sintéticos de opciones.

Acopla un solver de referencia (Black-Scholes o Heston) con un
:class:`~src.datasets.sampler.Sampler` y produce los ``OptionDataset``
que consumen los entrenamientos. Aquí se aplica además el filtro de no
arbitraje: las muestras cuyo precio cae fuera de las cotas teóricas se
descartan y se vuelven a sortear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.domain import Domain


class Sampler(Protocol):
    """Interfaz mínima que esperan los generadores."""

    def sample(self, n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
        ...


@dataclass(frozen=True)
class OptionDataset(Dataset):
    """Dataset de Torch con inputs normalizados y precios de call normalizados.

    Attributes:
        features: Inputs en ``[0, 1]^d`` listos para alimentar la red.
        prices: Precios de call normalizados ``C/K``.
        deltas: Deltas analíticas si el dataset las trae (solo training
            differencial). ``None`` en caso contrario.
        raw_inputs: Inputs en escala financiera; útiles para evaluar el
            solver de referencia sin re-denormalizar.
        input_names: Nombres de columnas en el mismo orden que
            ``features`` y ``raw_inputs``.
    """

    features: torch.Tensor
    prices: torch.Tensor
    deltas: torch.Tensor | None
    raw_inputs: torch.Tensor
    input_names: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "features": self.features[index],
            "price": self.prices[index],
            "raw_inputs": self.raw_inputs[index],
        }
        if self.deltas is not None:
            item["delta"] = self.deltas[index]
        return item


@dataclass(frozen=True)
class GeneratedBatch:
    """Resultado de una ronda de muestreo del :class:`DatasetGenerator`.

    Lleva, además de las muestras aceptadas, los conteos necesarios
    para diagnosticar la tasa de rechazo por no arbitraje.
    """

    raw_inputs: np.ndarray
    features: np.ndarray
    prices: np.ndarray
    deltas: np.ndarray | None
    attempted_count: int
    accepted_count: int
    rejected_count: int


@dataclass(frozen=True)
class DatasetGenerator:
    """Genera datasets sintéticos combinando un solver y un sampler.

    El generador sobre-muestrea (factor ``1.2``) y vuelve a sortear si
    quedan huecos tras filtrar por no arbitraje, hasta diez rondas. Si
    al cabo no se completa ``n_samples``, lanza ``RuntimeError``.

    Attributes:
        solver: Pricer de referencia (``BlackScholesSolver`` o
            ``HestonSolver``).
        domain: Hipercubo de entrada y normalización min-max.
        sampler: Estrategia de muestreo del dominio.
        model_family: ``"black_scholes"`` o ``"heston"``; selecciona la
            firma del solver al pedir precio (y, si aplica, Delta).
        include_delta: Si ``True``, computa y guarda la Delta analítica
            para entrenamientos diferenciales (E5).
        strike: Strike fijo del proyecto (``K = 1``).
        no_arbitrage_tolerance: Holgura para aceptar precios y Deltas
            cercanos a las cotas teóricas.
        dtype: dtype de los tensores resultantes.
    """

    solver: object
    domain: Domain
    sampler: Sampler
    model_family: str
    include_delta: bool = False
    strike: float = 1.0
    no_arbitrage_tolerance: float = 1e-7
    dtype: torch.dtype = torch.float32

    def generate(self, n_samples: int, seed: int | None = None) -> OptionDataset:
        """Devuelve un dataset con exactamente ``n_samples`` puntos válidos."""
        if n_samples <= 0:
            raise ValueError("n_samples must be strictly positive")

        rng = np.random.default_rng(seed)
        raw_parts: list[np.ndarray] = []
        price_parts: list[np.ndarray] = []
        delta_parts: list[np.ndarray] = []
        remaining = n_samples

        for _ in range(10):
            draw_count = max(remaining, int(np.ceil(1.2 * remaining)))
            batch = self.generate_batch(draw_count, rng)
            accepted = min(remaining, batch.accepted_count)
            if accepted > 0:
                raw_parts.append(batch.raw_inputs[:accepted])
                price_parts.append(batch.prices[:accepted])
                if batch.deltas is not None:
                    delta_parts.append(batch.deltas[:accepted])
                remaining -= accepted
            if remaining == 0:
                break

        if remaining > 0:
            raise RuntimeError("could not generate enough valid option samples")

        raw_inputs = np.concatenate(raw_parts, axis=0)
        prices = np.concatenate(price_parts, axis=0)
        deltas = None if not delta_parts else np.concatenate(delta_parts, axis=0)

        return self.to_dataset(raw_inputs, prices, deltas)

    def generate_batch(self, draw_count: int, rng: np.random.Generator) -> GeneratedBatch:
        """Una sola ronda de sampleo + pricing + filtro de no arbitraje."""
        if draw_count <= 0:
            raise ValueError("draw_count must be strictly positive")

        raw_batch = self.sampler.sample(draw_count, rng=rng)
        price_batch, delta_batch = self._price(raw_batch)
        valid_mask = self._valid_mask(raw_batch, price_batch, delta_batch)
        raw_inputs = raw_batch[valid_mask]
        prices = price_batch[valid_mask]
        deltas = None if delta_batch is None else delta_batch[valid_mask]
        return GeneratedBatch(
            raw_inputs=raw_inputs,
            features=self.domain.normalize(raw_inputs),
            prices=prices,
            deltas=deltas,
            attempted_count=draw_count,
            accepted_count=int(valid_mask.sum()),
            rejected_count=int(draw_count - valid_mask.sum()),
        )

    def to_dataset(
        self,
        raw_inputs: np.ndarray,
        prices: np.ndarray,
        deltas: np.ndarray | None = None,
    ) -> OptionDataset:
        features = self.domain.normalize(raw_inputs)
        return OptionDataset(
            features=torch.as_tensor(features, dtype=self.dtype),
            prices=torch.as_tensor(prices[:, None], dtype=self.dtype),
            deltas=None if deltas is None else torch.as_tensor(deltas[:, None], dtype=self.dtype),
            raw_inputs=torch.as_tensor(raw_inputs, dtype=self.dtype),
            input_names=self.domain.input_names,
        )

    def _price(self, raw_inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if self.model_family == "black_scholes":
            prices = self.solver.call_price(
                spot=raw_inputs[:, 0],
                strike=self.strike,
                maturity=raw_inputs[:, 1],
                rate=raw_inputs[:, 2],
                volatility=raw_inputs[:, 3],
                dividend_yield=self.domain.dividend_yield,
            )
            deltas = None
            if self.include_delta:
                deltas = self.solver.call_delta(
                    spot=raw_inputs[:, 0],
                    strike=self.strike,
                    maturity=raw_inputs[:, 1],
                    rate=raw_inputs[:, 2],
                    volatility=raw_inputs[:, 3],
                    dividend_yield=self.domain.dividend_yield,
                )
        elif self.model_family == "heston":
            if self.include_delta:
                prices, deltas = self.solver.call_price_and_delta(
                    spot=raw_inputs[:, 0],
                    strike=self.strike,
                    maturity=raw_inputs[:, 1],
                    rate=raw_inputs[:, 2],
                    v0=raw_inputs[:, 3],
                    theta=raw_inputs[:, 4],
                    kappa=raw_inputs[:, 5],
                    xi=raw_inputs[:, 6],
                    rho=raw_inputs[:, 7],
                    dividend_yield=self.domain.dividend_yield,
                )
            else:
                prices = self.solver.call_price(
                    spot=raw_inputs[:, 0],
                    strike=self.strike,
                    maturity=raw_inputs[:, 1],
                    rate=raw_inputs[:, 2],
                    v0=raw_inputs[:, 3],
                    theta=raw_inputs[:, 4],
                    kappa=raw_inputs[:, 5],
                    xi=raw_inputs[:, 6],
                    rho=raw_inputs[:, 7],
                    dividend_yield=self.domain.dividend_yield,
                )
                deltas = None
        else:
            raise ValueError("model_family must be 'black_scholes' or 'heston'")

        return np.asarray(prices, dtype=float), None if deltas is None else np.asarray(deltas, dtype=float)

    def _valid_mask(
        self,
        raw_inputs: np.ndarray,
        prices: np.ndarray,
        deltas: np.ndarray | None,
    ) -> np.ndarray:
        moneyness = raw_inputs[:, 0]
        maturity = raw_inputs[:, 1]
        rate = raw_inputs[:, 2]
        discount = np.exp(-rate * maturity)
        lower = np.maximum(moneyness - self.strike * discount, 0.0)
        upper = moneyness
        valid = (
            np.isfinite(prices)
            & (prices >= lower - self.no_arbitrage_tolerance)
            & (prices <= upper + self.no_arbitrage_tolerance)
        )
        if deltas is not None:
            valid &= np.isfinite(deltas) & (deltas >= -self.no_arbitrage_tolerance)
            valid &= deltas <= 1.0 + self.no_arbitrage_tolerance
        return valid
