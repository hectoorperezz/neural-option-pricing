"""Pricer Black-Scholes para calls europeas.

Devuelve precio, Delta y Vega en forma cerrada. Sirve como ground truth
del proyecto: los surrogates de Black-Scholes se contrastan contra
estos valores exactos antes de pasar a Heston (ver ``docs/tasks.md``,
Fase 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf
from typing import Any
import warnings

import numpy as np

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from scipy.special import ndtr as scipy_normal_cdf
except Exception:  # pragma: no cover - depends on the local SciPy installation
    scipy_normal_cdf = None


SQRT_2PI = np.sqrt(2.0 * np.pi)
SQRT_2 = np.sqrt(2.0)


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / SQRT_2PI


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    """Φ(x) usando SciPy si está disponible; fallback a ``erf`` vectorizado."""
    if scipy_normal_cdf is not None:
        return scipy_normal_cdf(x)
    return 0.5 * (1.0 + np.vectorize(erf, otypes=[float])(x / SQRT_2))


def _to_output(value: np.ndarray) -> float | np.ndarray:
    """Colapsa arrays 0-d a ``float`` para no propagar arrays escalares."""
    if value.ndim == 0:
        return float(value)
    return value


@dataclass(frozen=True)
class BlackScholesSolver:
    """Solver analítico Black-Scholes-Merton para calls europeas.

    Cada método acepta escalares o arrays y devuelve el mismo tipo. Los
    parámetros financieros son los habituales: ``spot``, ``strike``,
    ``maturity`` (años), ``rate``, ``volatility`` y ``dividend_yield``
    (por convención del proyecto, ``q = 0``).
    """

    def call_price(
        self,
        spot: Any,
        strike: Any,
        maturity: Any,
        rate: Any,
        volatility: Any,
        dividend_yield: Any = 0.0,
    ) -> float | np.ndarray:
        """Precio de la call: ``S e^{-qT} Φ(d_1) - K e^{-rT} Φ(d_2)``."""
        spot_arr, strike_arr, maturity_arr, rate_arr, vol_arr, q_arr = self._as_arrays(
            spot, strike, maturity, rate, volatility, dividend_yield
        )
        d1, d2 = self._d1_d2(spot_arr, strike_arr, maturity_arr, rate_arr, vol_arr, q_arr)
        price = (
            spot_arr * np.exp(-q_arr * maturity_arr) * _normal_cdf(d1)
            - strike_arr * np.exp(-rate_arr * maturity_arr) * _normal_cdf(d2)
        )
        return _to_output(price)

    def call_delta(
        self,
        spot: Any,
        strike: Any,
        maturity: Any,
        rate: Any,
        volatility: Any,
        dividend_yield: Any = 0.0,
    ) -> float | np.ndarray:
        """Delta de la call: ``e^{-qT} Φ(d_1)``."""
        spot_arr, strike_arr, maturity_arr, rate_arr, vol_arr, q_arr = self._as_arrays(
            spot, strike, maturity, rate, volatility, dividend_yield
        )
        d1, _ = self._d1_d2(spot_arr, strike_arr, maturity_arr, rate_arr, vol_arr, q_arr)
        delta = np.exp(-q_arr * maturity_arr) * _normal_cdf(d1)
        return _to_output(delta)

    def call_vega(
        self,
        spot: Any,
        strike: Any,
        maturity: Any,
        rate: Any,
        volatility: Any,
        dividend_yield: Any = 0.0,
    ) -> float | np.ndarray:
        """Vega de la call: ``S e^{-qT} \\sqrt{T}\\,\\varphi(d_1)``."""
        spot_arr, strike_arr, maturity_arr, rate_arr, vol_arr, q_arr = self._as_arrays(
            spot, strike, maturity, rate, volatility, dividend_yield
        )
        d1, _ = self._d1_d2(spot_arr, strike_arr, maturity_arr, rate_arr, vol_arr, q_arr)
        vega = spot_arr * np.exp(-q_arr * maturity_arr) * np.sqrt(maturity_arr) * _normal_pdf(d1)
        return _to_output(vega)

    def _d1_d2(
        self,
        spot: np.ndarray,
        strike: np.ndarray,
        maturity: np.ndarray,
        rate: np.ndarray,
        volatility: np.ndarray,
        dividend_yield: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calcula ``d_1`` y ``d_2`` con validación de inputs."""
        self._validate_inputs(spot, strike, maturity, volatility)
        vol_sqrt_t = volatility * np.sqrt(maturity)
        d1 = (
            np.log(spot / strike)
            + (rate - dividend_yield + 0.5 * volatility * volatility) * maturity
        ) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        return d1, d2

    def _as_arrays(self, *values: Any) -> tuple[np.ndarray, ...]:
        """Convierte cada entrada a ``ndarray`` y aplica broadcasting común."""
        broadcast = np.broadcast_arrays(*[np.asarray(value, dtype=float) for value in values])
        return tuple(broadcast)

    def _validate_inputs(
        self,
        spot: np.ndarray,
        strike: np.ndarray,
        maturity: np.ndarray,
        volatility: np.ndarray,
    ) -> None:
        if np.any(spot <= 0.0):
            raise ValueError("spot must be strictly positive")
        if np.any(strike <= 0.0):
            raise ValueError("strike must be strictly positive")
        if np.any(maturity <= 0.0):
            raise ValueError("maturity must be strictly positive")
        if np.any(volatility <= 0.0):
            raise ValueError("volatility must be strictly positive")
