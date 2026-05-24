from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.solvers.black_scholes import BlackScholesSolver


@dataclass(frozen=True)
class ImpliedVolatilityInverter:
    """Black-Scholes implied-volatility inverter for European calls."""

    solver: BlackScholesSolver = BlackScholesSolver()
    min_volatility: float = 1e-8
    max_volatility: float = 5.0
    tolerance: float = 1e-10
    max_iterations: int = 100

    def solve_call(
        self,
        price: float,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend_yield: float = 0.0,
        initial_guess: float = 0.2,
    ) -> float:
        self._validate_inputs(price, spot, strike, maturity, rate, dividend_yield)

        lower = self._lower_bound(spot, strike, maturity, rate, dividend_yield)
        upper = spot * np.exp(-dividend_yield * maturity)
        if price < lower - self.tolerance or price > upper + self.tolerance:
            raise ValueError("price is outside no-arbitrage bounds for a call option")
        if abs(price - lower) <= self.tolerance:
            return self.min_volatility

        sigma = float(np.clip(initial_guess, self.min_volatility, self.max_volatility))
        for _ in range(self.max_iterations):
            model_price = self.solver.call_price(
                spot, strike, maturity, rate, sigma, dividend_yield
            )
            diff = model_price - price
            if abs(diff) <= self.tolerance:
                return sigma

            vega = self.solver.call_vega(spot, strike, maturity, rate, sigma, dividend_yield)
            if vega <= 1e-12:
                break

            next_sigma = sigma - diff / vega
            if not np.isfinite(next_sigma) or not (
                self.min_volatility <= next_sigma <= self.max_volatility
            ):
                break
            sigma = float(next_sigma)

        return self._bisect(price, spot, strike, maturity, rate, dividend_yield)

    def _bisect(
        self,
        price: float,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend_yield: float,
    ) -> float:
        low = self.min_volatility
        high = self.max_volatility

        high_price = self.solver.call_price(spot, strike, maturity, rate, high, dividend_yield)
        if high_price < price - self.tolerance:
            raise ValueError("price cannot be matched within volatility bounds")

        for _ in range(self.max_iterations):
            mid = 0.5 * (low + high)
            mid_price = self.solver.call_price(spot, strike, maturity, rate, mid, dividend_yield)
            diff = mid_price - price
            if abs(diff) <= self.tolerance:
                return mid
            if diff > 0.0:
                high = mid
            else:
                low = mid
        return 0.5 * (low + high)

    def _validate_inputs(
        self,
        price: float,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend_yield: float,
    ) -> None:
        if spot <= 0.0:
            raise ValueError("spot must be strictly positive")
        if strike <= 0.0:
            raise ValueError("strike must be strictly positive")
        if maturity <= 0.0:
            raise ValueError("maturity must be strictly positive")
        if price < 0.0:
            raise ValueError("price must be non-negative")
        if not np.isfinite(rate) or not np.isfinite(dividend_yield):
            raise ValueError("rate and dividend_yield must be finite")

    def _lower_bound(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend_yield: float,
    ) -> float:
        forward_intrinsic = (
            spot * np.exp(-dividend_yield * maturity) - strike * np.exp(-rate * maturity)
        )
        return max(forward_intrinsic, 0.0)
