from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.integrate import quad


def _to_output(value: np.ndarray) -> float | np.ndarray:
    if value.ndim == 0:
        return float(value)
    return value


@dataclass(frozen=True)
class HestonSolver:
    """Fourier pricer for European calls under the Heston model."""

    integration_upper_bound: float = 100.0
    integration_lower_bound: float = 1e-8
    absolute_tolerance: float = 1e-8
    relative_tolerance: float = 1e-8
    quad_limit: int = 200
    no_arbitrage_tolerance: float = 1e-6

    def call_price(
        self,
        spot: Any,
        strike: Any,
        maturity: Any,
        rate: Any,
        v0: Any,
        theta: Any,
        kappa: Any,
        xi: Any,
        rho: Any,
        dividend_yield: Any = 0.0,
    ) -> float | np.ndarray:
        arrays = self._as_arrays(
            spot, strike, maturity, rate, v0, theta, kappa, xi, rho, dividend_yield
        )
        prices = np.empty_like(arrays[0], dtype=float)

        for index in np.ndindex(prices.shape):
            p1, p2 = self._call_probabilities_scalar(*[float(value[index]) for value in arrays])
            s = float(arrays[0][index])
            k = float(arrays[1][index])
            t = float(arrays[2][index])
            r = float(arrays[3][index])
            q = float(arrays[9][index])
            raw_price = s * np.exp(-q * t) * p1 - k * np.exp(-r * t) * p2
            prices[index] = self._apply_call_bounds(raw_price, s, k, t, r, q)

        return _to_output(prices)

    def call_delta(
        self,
        spot: Any,
        strike: Any,
        maturity: Any,
        rate: Any,
        v0: Any,
        theta: Any,
        kappa: Any,
        xi: Any,
        rho: Any,
        dividend_yield: Any = 0.0,
    ) -> float | np.ndarray:
        arrays = self._as_arrays(
            spot, strike, maturity, rate, v0, theta, kappa, xi, rho, dividend_yield
        )
        deltas = np.empty_like(arrays[0], dtype=float)

        for index in np.ndindex(deltas.shape):
            p1 = self._call_probability_scalar(1, *[float(value[index]) for value in arrays])
            t = float(arrays[2][index])
            q = float(arrays[9][index])
            deltas[index] = np.exp(-q * t) * p1

        return _to_output(deltas)

    def call_probabilities(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        v0: float,
        theta: float,
        kappa: float,
        xi: float,
        rho: float,
        dividend_yield: float = 0.0,
    ) -> tuple[float, float]:
        return self._call_probabilities_scalar(
            spot, strike, maturity, rate, v0, theta, kappa, xi, rho, dividend_yield
        )

    def _call_probabilities_scalar(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        v0: float,
        theta: float,
        kappa: float,
        xi: float,
        rho: float,
        dividend_yield: float,
    ) -> tuple[float, float]:
        self._validate_inputs(spot, strike, maturity, v0, theta, kappa, xi, rho)
        log_strike = np.log(strike)
        forward_spot = spot * np.exp((rate - dividend_yield) * maturity)

        def p1_integrand(u: float) -> float:
            shifted_cf = self._characteristic_function(
                u - 1j, spot, maturity, rate, dividend_yield, v0, theta, kappa, xi, rho
            )
            value = np.exp(-1j * u * log_strike) * shifted_cf / (1j * u * forward_spot)
            return float(np.real(value))

        def p2_integrand(u: float) -> float:
            cf = self._characteristic_function(
                u, spot, maturity, rate, dividend_yield, v0, theta, kappa, xi, rho
            )
            value = np.exp(-1j * u * log_strike) * cf / (1j * u)
            return float(np.real(value))

        p1_integral = self._integrate(p1_integrand)
        p2_integral = self._integrate(p2_integrand)
        return 0.5 + p1_integral / np.pi, 0.5 + p2_integral / np.pi

    def _call_probability_scalar(
        self,
        probability_index: int,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        v0: float,
        theta: float,
        kappa: float,
        xi: float,
        rho: float,
        dividend_yield: float,
    ) -> float:
        if probability_index not in (1, 2):
            raise ValueError("probability_index must be 1 or 2")
        self._validate_inputs(spot, strike, maturity, v0, theta, kappa, xi, rho)
        log_strike = np.log(strike)
        forward_spot = spot * np.exp((rate - dividend_yield) * maturity)

        def integrand(u: float) -> float:
            if probability_index == 1:
                cf = self._characteristic_function(
                    u - 1j,
                    spot,
                    maturity,
                    rate,
                    dividend_yield,
                    v0,
                    theta,
                    kappa,
                    xi,
                    rho,
                )
                value = np.exp(-1j * u * log_strike) * cf / (1j * u * forward_spot)
            elif probability_index == 2:
                cf = self._characteristic_function(
                    u,
                    spot,
                    maturity,
                    rate,
                    dividend_yield,
                    v0,
                    theta,
                    kappa,
                    xi,
                    rho,
                )
                value = np.exp(-1j * u * log_strike) * cf / (1j * u)
            return float(np.real(value))

        return 0.5 + self._integrate(integrand) / np.pi

    def _apply_call_bounds(
        self,
        price: float,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend_yield: float,
    ) -> float:
        lower = max(
            spot * np.exp(-dividend_yield * maturity) - strike * np.exp(-rate * maturity),
            0.0,
        )
        upper = spot * np.exp(-dividend_yield * maturity)
        if lower - self.no_arbitrage_tolerance <= price < lower:
            return lower
        if upper < price <= upper + self.no_arbitrage_tolerance:
            return upper
        return price

    def _integrate(self, integrand: Callable[[float], float]) -> float:
        value, _ = quad(
            integrand,
            self.integration_lower_bound,
            self.integration_upper_bound,
            epsabs=self.absolute_tolerance,
            epsrel=self.relative_tolerance,
            limit=self.quad_limit,
        )
        return float(value)

    def _characteristic_function(
        self,
        u: complex,
        spot: float,
        maturity: float,
        rate: float,
        dividend_yield: float,
        v0: float,
        theta: float,
        kappa: float,
        xi: float,
        rho: float,
    ) -> complex:
        iu = 1j * u
        variance_scale = xi * xi
        drift = rate - dividend_yield
        log_spot = np.log(spot)

        d = np.sqrt((rho * xi * iu - kappa) ** 2 + variance_scale * (iu + u * u))
        numerator = kappa - rho * xi * iu - d
        denominator = kappa - rho * xi * iu + d
        g = numerator / denominator
        exp_minus_dt = np.exp(-d * maturity)

        c = (
            iu * (log_spot + drift * maturity)
            + kappa
            * theta
            / variance_scale
            * (numerator * maturity - 2.0 * np.log((1.0 - g * exp_minus_dt) / (1.0 - g)))
        )
        d_term = numerator / variance_scale * (1.0 - exp_minus_dt) / (1.0 - g * exp_minus_dt)
        return complex(np.exp(c + d_term * v0))

    def _as_arrays(self, *values: Any) -> tuple[np.ndarray, ...]:
        broadcast = np.broadcast_arrays(*[np.asarray(value, dtype=float) for value in values])
        return tuple(broadcast)

    def _validate_inputs(
        self,
        spot: float,
        strike: float,
        maturity: float,
        v0: float,
        theta: float,
        kappa: float,
        xi: float,
        rho: float,
    ) -> None:
        if spot <= 0.0:
            raise ValueError("spot must be strictly positive")
        if strike <= 0.0:
            raise ValueError("strike must be strictly positive")
        if maturity <= 0.0:
            raise ValueError("maturity must be strictly positive")
        if v0 < 0.0:
            raise ValueError("v0 must be non-negative")
        if theta <= 0.0:
            raise ValueError("theta must be strictly positive")
        if kappa <= 0.0:
            raise ValueError("kappa must be strictly positive")
        if xi <= 0.0:
            raise ValueError("xi must be strictly positive")
        if not -1.0 < rho < 1.0:
            raise ValueError("rho must be between -1 and 1")
